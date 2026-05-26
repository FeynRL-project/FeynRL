import torch
from torch.utils.data import Dataset, DataLoader
import os
from datasets import load_dataset

class PairedFeed(Dataset):
    '''
        This is a general dataset to handle prompt and answer pairs.
        The data should be in a parquet format and system prompt is optional.
    '''
    def __init__(self, 
                prompt_key,
                answer_key, 
                max_seq_len,
                tokenizer=None, 
                data_path="",
                ):
        assert prompt_key != "", "prompt_key cannot be empty"
        assert answer_key != "", "answer_key cannot be empty"
        assert max_seq_len > 0, "max_seq_len must be greater than 0"
        assert tokenizer is not None, "tokenizer cannot be None"
        assert isinstance(data_path, str), "data_path must be a string"
        assert os.path.exists(os.path.expanduser(data_path)), f"{data_path} does not exist"

        # add this assert to make sure that tokenizer has a pad token (or if not,
        # we already added during loading)
        assert tokenizer.pad_token_id is not None, "tokenizer must have a pad token"
        assert tokenizer.eos_token_id is not None, "tokenizer must have an eos token"

        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.data_path = data_path
        # load data into cpu memory
        self._load_data()

    def _load_data(self):
        '''
           Loads the data from a parquet file.
        '''
        try:
            # This acts like a list but reads from disk/cache on demand.
            # if we don't use split, it will return a DatasetDict
            # DatasetDict({train: Dataset({...})})
            # if we use split=train, it will return a Dataset
            # Dataset({...})
            # split here doesn't mean our actual splits. it is just for compatibility with huggingface datasets.
            self.data = load_dataset("parquet", data_files=self.data_path, split="train")

        except Exception as e:
            raise Exception(f"Failed to load data from {self.data_path}: {str(e)}")

        self.len_data = len(self.data)

    def _process_answer(self, answer):
        '''
           Processes the answer to add EOS token.
        '''
        answer_ids_output = self.tokenizer(answer,
                                           return_tensors='pt',
                                           add_special_tokens=False)
        # manually add EOS token to cover cases where the answer doesn't end with a space,
        # the tokenizer might merge the last word and the EOS token into a
        # single unknown or different token.
        eos_tensor = torch.tensor([self.tokenizer.eos_token_id],dtype=answer_ids_output['input_ids'].dtype)
        answer_ids = torch.cat([answer_ids_output['input_ids'][0], eos_tensor], dim=0)
        # finally create the attention mask
        answer_attn_mask = torch.cat([answer_ids_output['attention_mask'][0],
                                      torch.tensor([1], dtype=answer_ids_output['attention_mask'].dtype)], dim=0)

        if len(answer_ids) <= 1:
            raise ValueError(f"Answer must tokenize to at least one token (excluding EOS). "
                             f"Got {len(answer_ids)} tokens total.")

        return answer_ids, answer_attn_mask

    def __getitem__(self, idx):
        '''
          Each example has the following format:
            current_sample = {"prompt": [{"role": "system", "content": "this is a system prompt"},
                                 {"role": "user", "content": "this is a user prompt step 1"},
                                 {"role": "assistant", "content": "this is an assistant response to step 1"},
                                 {"role": "user", "content": "this is a user prompt in step 2"},
                                 {"role": "assistant", "content": "this is an assistant response to step 2"},
                                 ...
                                 ],
                      "answer": "this is an answer",
            }
          Loss is computed on all assistant responses including the final answer.
        '''
        # Ensure native int: DataLoader/samplers may pass numpy.int64, which Dataset rejects
        idx = int(idx)
        current_sample = self.data[idx]

        if self.prompt_key not in current_sample:
            raise KeyError(f"Missing key '{self.prompt_key}' in sample {current_sample}: keys={list(current_sample.keys())}")

        if self.answer_key not in current_sample:
            raise KeyError(f"Missing key '{self.answer_key}' in sample {current_sample}: keys={list(current_sample.keys())}")

        message = current_sample[self.prompt_key]
        answer  = current_sample[self.answer_key]

        # message cannot be empty
        if not message or (isinstance(message, list) and len(message) == 0):
            raise ValueError(f"Sample {idx}:{current_sample}: Prompt/message cannot be empty")

        # answer cannot be empty
        if not answer or (isinstance(answer, str) and answer.strip() == ""):
            raise ValueError(f"Sample {current_sample}: Answer cannot be empty or whitespace-only")

        if len(message) == 1 or (len(message) == 2 and message[0]['role'] == 'system' and message[1]['role'] == 'user'):
            return self._get_single_turn(idx, message, answer)

        else:
            return self._get_multi_turns(idx, message, answer)

    def _get_single_turn(self, idx, message, answer):
        '''
           Handles a single turn of the conversation.
        '''
        # 1. Tokenize the prompt
        # When tokenize=True and return_tensors='pt', it returns shape [1, seq_len]
        # [0]: [1, seq_len] -> [seq_len]
        prompt_ids = self.tokenizer.apply_chat_template(conversation=message,
                                                        add_generation_prompt=True,
                                                        tokenize=True,
                                                        return_tensors='pt'
                                                        )[0]
        prompt_attn_mask = torch.ones_like(prompt_ids)
        prompt_len = len(prompt_ids)

        # 2. Validate prompt length
        if prompt_len >= self.max_seq_len or prompt_len == 0:
            raise ValueError(f"Prompt in sample {idx}:{message}: too long or empty: "
                             f"prompt must be at most {self.max_seq_len} tokens (got {prompt_len})")

        # 3. Tokenize answer + add EOS
        answer_ids, answer_attn_mask = self._process_answer(answer)

        # 4. Build sequence
        seq_ids = torch.cat((prompt_ids, answer_ids), dim=-1).to(dtype=torch.long)
        seq_attn_mask = torch.cat((prompt_attn_mask, answer_attn_mask), dim=-1)
        total_seq_len = len(seq_ids)

        # 5. Validate minimum sequence length
        if total_seq_len < 2:
            raise ValueError(f"Sequence too short: prompt + answer must be at least 2 tokens (got {total_seq_len})")

        # 6. length check
        if total_seq_len > self.max_seq_len:
            # this should be ideally handled in data-preprocessing step
            # we might lose the EOS token here. This is acceptable in SFT training though
            # as the model learns max length reached.
            seq_ids = seq_ids[:self.max_seq_len]
            seq_attn_mask = seq_attn_mask[:self.max_seq_len]
            total_seq_len = len(seq_ids)

            answer_start_idx = prompt_len
            answer_end_idx = self.max_seq_len
            actual_answer_tokens_in_seq = answer_end_idx - answer_start_idx

            # We need at least 1 answer token to train on
            if actual_answer_tokens_in_seq < 1:
                raise ValueError(f"{message}: After truncation, no answer tokens remain.")

        # 7. pad if necessary
        elif total_seq_len < self.max_seq_len:
            padding_len = self.max_seq_len - total_seq_len

            # add padding tokens to ids
            padding_tokens = torch.full((padding_len,), self.tokenizer.pad_token_id, dtype=seq_ids.dtype)
            seq_ids = torch.cat((seq_ids, padding_tokens), dim=-1)

            # add zeros to attention mask as padding
            padding_attn_mask = torch.zeros(size=(padding_len,), dtype=seq_attn_mask.dtype)
            seq_attn_mask = torch.cat((seq_attn_mask, padding_attn_mask), dim=-1)

        # Labels are created by shifting seq_ids by one position, so they'll have length T-1.
        # Therefore, the loss mask must also be of shape [T-1].
        # We don't need to worry about padding tokens as they are already handled by seq_attn_mask if any (pads are zero).
        loss_mask = seq_attn_mask[1:].clone()

        # Mask out prompt tokens.
        # Since labels are shifted by one position, the prompt appears in indices
        # [:len(prompt_ids) - 1] in the label sequence (not [:len(prompt_ids)]).
        if prompt_len > 1:
            loss_mask[:prompt_len - 1] = 0

        # After masking, we should have at least 1 unmasked answer token
        if loss_mask.sum().item() == 0:
            raise ValueError(f"Sample {idx}:{message}: No training tokens left after masking "
                         f"Prompt length: {len(prompt_ids)}, Answer length: {len(answer_ids)}, "
                         f"Total length: {total_seq_len}.")

        return {
            "input_ids": seq_ids, # T
            "attn_mask": seq_attn_mask, # T
            "loss_mask": loss_mask, # T-1
        }

    def _get_multi_turns(self, idx, message, answer):
        '''
            Handles multi-turn conversations
        '''
        # 1. Tokenize incrementally to track assistant response boundaries to handle loss masking
        # We need to find where each assistant response starts/ends in the token sequence
        assistant_ranges = []  # list of (start_idx, end_idx) for assistant content
        current_len = 0

        for i, turn in enumerate(message):
            # Tokenize conversation up to and including this turn
            conversation_so_far = message[:i + 1]
            # When tokenize=True and return_tensors='pt', it returns shape [1, seq_len]
            # [0]: [1, seq_len] -> [seq_len]
            tokens_so_far = self.tokenizer.apply_chat_template(conversation=conversation_so_far,
                                                               add_generation_prompt=False, # no generation prompt for intermediate turns
                                                               tokenize=True,
                                                               return_tensors='pt'
                                                               )[0]
            new_len = len(tokens_so_far)

            if turn.get("role") == "assistant":
                # This range corresponds to the assistant's response (including any formatting)
                assistant_ranges.append((current_len, new_len))

            current_len = new_len

        # 2. Tokenize the full prompt with generation prompt for the final answer
        prompt_ids = self.tokenizer.apply_chat_template(conversation=message,
                                                        add_generation_prompt=True, # add generation prompt for final answer
                                                        tokenize=True,
                                                        return_tensors='pt'
                                                        )[0]
        prompt_attn_mask = torch.ones_like(prompt_ids)
        prompt_len       = len(prompt_ids)

        # 3. Validate prompt length
        if prompt_len >= self.max_seq_len or prompt_len == 0:
            raise ValueError(f"Prompt in sample {idx}:{message}: too long or empty: "
                             f"prompt must be at most {self.max_seq_len} tokens (got {prompt_len})")

        # 4. Tokenize answer + add EOS
        answer_ids, answer_attn_mask = self._process_answer(answer)

        # Add final answer range (starts at prompt_len, ends at prompt_len + answer_len)
        assistant_ranges.append((prompt_len, prompt_len + len(answer_ids)))

        seq_ids = torch.cat((prompt_ids, answer_ids), dim=-1).to(dtype=torch.long)
        seq_attn_mask = torch.cat((prompt_attn_mask, answer_attn_mask), dim=-1)
        total_seq_len = len(seq_ids)

        # 5. Validate minimum sequence length
        if total_seq_len < 2:
            raise ValueError(f"Sequence too short: prompt + answer must be at least 2 tokens (got {total_seq_len})")

        # 6. Length check - truncate if necessary
        if total_seq_len > self.max_seq_len:
            seq_ids = seq_ids[:self.max_seq_len]
            seq_attn_mask = seq_attn_mask[:self.max_seq_len]
            total_seq_len = len(seq_ids)

            # Adjust assistant ranges for truncation
            adjusted_ranges = []
            for start, end in assistant_ranges:
                if start < self.max_seq_len:
                    adjusted_ranges.append((start, min(end, self.max_seq_len)))
            assistant_ranges = adjusted_ranges

            # Check we have at least some assistant tokens after truncation
            total_assistant_tokens = sum(end - start for start, end in assistant_ranges)
            if total_assistant_tokens < 1:
                raise ValueError(f"Sample {idx}:{message}: After truncation, no assistant tokens remain.")

        # 7. Pad if necessary
        elif total_seq_len < self.max_seq_len:
            padding_len = self.max_seq_len - total_seq_len

            padding_tokens = torch.full((padding_len,), self.tokenizer.pad_token_id, dtype=seq_ids.dtype)
            seq_ids = torch.cat((seq_ids, padding_tokens), dim=-1)

            padding_attn_mask = torch.zeros(size=(padding_len,), dtype=seq_attn_mask.dtype)
            seq_attn_mask = torch.cat((seq_attn_mask, padding_attn_mask), dim=-1)

        # 8. Build loss mask for multi-turn
        # Loss mask has shape [T-1] because labels are shifted by one position
        loss_mask = torch.zeros(self.max_seq_len - 1, dtype=seq_attn_mask.dtype)

        # For each assistant range, mark the corresponding positions in loss_mask
        # Since labels are shifted, token at position i predicts token at position i+1
        # So for assistant content at positions [start, end), we train on labels [start, end-1]
        # which means loss_mask positions [start-1, end-2] should be 1...
        # Actually: label[i] = token[i+1], so to train on token[j], we need loss_mask[j-1] = 1
        for start, end in assistant_ranges:
            # We want to predict tokens in range [start, end)
            # In the shifted label sequence, these correspond to loss_mask indices [start-1, end-1)
            loss_start = max(0, start - 1)
            loss_end   = min(end - 1, self.max_seq_len - 1)
            if loss_start < loss_end:
                loss_mask[loss_start:loss_end] = 1

        # Apply attention mask to loss mask (don't compute loss on padding)
        loss_mask = loss_mask * seq_attn_mask[1:]

        # After masking, we should have at least 1 unmasked token
        if loss_mask.sum().item() == 0:
            raise ValueError(f"Sample {idx}:{message}: No training tokens left after masking "
                         f"Total length: {total_seq_len}.")

        return {
            "input_ids": seq_ids, # T
            "attn_mask": seq_attn_mask, # T
            "loss_mask": loss_mask, # T-1
        }

    def __len__(self):
        return self.len_data

if __name__ == "__main__":
    '''
        This is a simple test to make sure the dataset works.
    '''
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    import pandas as pd

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
    # add pad token if it doesn't exist, not useful here but good practice
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # this is an example of how the data should look like
    random_prompts = [
                   {'prompt': [{"role": "system", "content": "You are a helpful assistant."},
                               {"role": "user", "content": "Hello, how are you?"}],
                    'answer': "I'm good, thanks!"
                   },
                   {'prompt': [{"role": "user", "content": "What is the meaning of life?"}],
                    'answer': "The meaning of life is 2000002."
                   },
                   {'prompt': [{"role": "user", "content": "What is the meaning of the universe?"}],
                    'answer': "The meaning of the universe is galaxy plus 2."
                   },
                   {'prompt': [{"role": "user", "content": "This is is a just rather long prompt that is going to be tokenized. This is a test to make sure the dataset works."}],
                    'answer': "This is a test to make sure the dataset works."
                   },
                    {'prompt': [{"role": "system", "content": "You are a concise assistant."},
                                {"role": "user", "content": "Give me a weird two-word nickname."},
                                {"role": "assistant", "content": "Neon Pickle."},
                                {"role": "user", "content": "Now give a different one."}],
                    'answer': "Velvet Comet."
                    },
                    {'prompt': [{"role": "user", "content": "Random fact, but fake."},
                                {"role": "assistant", "content": "Otters invented the first toast."},
                                {"role": "user", "content": "Another fake fact."}],
                    'answer': "Clouds are just shy mountains."
                    }

                   ]
    df = pd.DataFrame(random_prompts)
    df.to_parquet("./promptonly.parquet", index=False)

    dataset = PairedFeed(
                    prompt_key="prompt",
                    answer_key="answer",
                    tokenizer=tokenizer,
                    max_seq_len=200,
                    data_path="./promptonly.parquet",
                      )
    dataloader = DataLoader(dataset, batch_size=3)
    for d in dataloader:
        print(d)

    from mixed_sampler import create_dataset_and_sampler
    dataset, sampler = create_dataset_and_sampler(data_paths=["./promptonly.parquet"],
                                                  prompt_key="prompt",
                                                  answer_key="answer",
                                                  max_seq_len=200,
                                                  tokenizer=tokenizer,
                                                  train_ratios={"promptonly":1},
                                                  split="train",
                                                  rank=0,
                                                  world_size=1,
                                                  seed=42,
                                                  local_batch_size=3,
                                                  dataset_cls=PairedFeed,
                                                  steps_per_epoch=100,
                                                  shuffle_within_batch=True,
                                                  dynamic_ratio_every_step=False)

    dummy_loader = DataLoader(dataset=dataset,
                              batch_sampler=sampler,
                              num_workers=0,
                              pin_memory=True,
                              worker_init_fn=None)
    for d in dummy_loader:
        print(d)
