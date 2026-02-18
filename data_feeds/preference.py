import torch
from torch.utils.data import Dataset, DataLoader
import os
from datasets import load_dataset

class PreferenceFeed(Dataset):
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
        self.chosen_key = answer_key
        self.rejected_key = "rejected_" + self.chosen_key
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
                      "rejected_answer": "this is a rejected answer",
            }
          Loss is computed on all assistant responses including the final answer.
        '''
        # Ensure native int: DataLoader/samplers may pass numpy.int64, which Dataset rejects
        idx = int(idx)
        current_sample = self.data[idx]

        if self.prompt_key not in current_sample:
            raise KeyError(f"Missing key '{self.prompt_key}' in sample {current_sample}: keys={list(current_sample.keys())}")

        if self.chosen_key not in current_sample:
            raise KeyError(f"Missing key '{self.chosen_key}' in sample {current_sample}: keys={list(current_sample.keys())}")

        if self.rejected_key not in current_sample:
            raise KeyError(f"Missing key '{self.rejected_key}' in sample {current_sample}: keys={list(current_sample.keys())}")


        message = current_sample[self.prompt_key]
        chosen_answer  = current_sample[self.chosen_key]
        rejected_answer = current_sample[self.rejected_key]

        # message cannot be empty
        if not message or (isinstance(message, list) and len(message) == 0):
            raise ValueError(f"Sample {idx}:{current_sample}: Prompt/message cannot be empty")

        # answer cannot be empty
        if not chosen_answer or (isinstance(chosen_answer, str) and chosen_answer.strip() == ""):
            raise ValueError(f"Sample {current_sample}: Answer cannot be empty or whitespace-only")

        if not rejected_answer or (isinstance(rejected_answer, str) and rejected_answer.strip() == ""):
            raise ValueError(f"Sample {current_sample}: Answer cannot be empty or whitespace-only")

        return self._get_sample(idx, message, chosen_answer, rejected_answer)

    def _get_sample(self, idx, message, chosen_answer, rejected_answer):
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
        chosen_answer_ids, chosen_answer_attn_mask = self._process_answer(chosen_answer)
        rejected_answer_ids, rejected_answer_attn_mask = self._process_answer(rejected_answer)

        # 4. Build sequence
        chosen_seq_ids = torch.cat((prompt_ids, chosen_answer_ids), dim=-1).to(dtype=torch.long)
        chosen_seq_attn_mask = torch.cat((prompt_attn_mask, chosen_answer_attn_mask), dim=-1)
        chosen_total_seq_len = len(chosen_seq_ids)
        chosen_seq_ids, chosen_seq_attn_mask = self._check_seq(message=message,
                                                              prompt_len=prompt_len,
                                                              total_seq_len=chosen_total_seq_len,
                                                              seq_ids=chosen_seq_ids,
                                                              seq_attn_mask=chosen_seq_attn_mask)

        rejected_seq_ids = torch.cat((prompt_ids, rejected_answer_ids), dim=-1).to(dtype=torch.long)
        rejected_seq_attn_mask = torch.cat((prompt_attn_mask, rejected_answer_attn_mask), dim=-1)
        rejected_total_seq_len = len(rejected_seq_ids)

        rejected_seq_ids, rejected_seq_attn_mask = self._check_seq(message=message,
                                                                  prompt_len=prompt_len,
                                                                  total_seq_len=rejected_total_seq_len,
                                                                  seq_ids=rejected_seq_ids,
                                                                  seq_attn_mask=rejected_seq_attn_mask)

        # Labels are created by shifting seq_ids by one position, so they'll have length T-1.
        # Therefore, the loss mask must also be of shape [T-1].
        # We don't need to worry about padding tokens as they are already handled by seq_attn_mask if any (pads are zero).
        chosen_loss_mask = chosen_seq_attn_mask[1:].clone()
        rejected_loss_mask = rejected_seq_attn_mask[1:].clone()

        # Mask out prompt tokens.
        # Since labels are shifted by one position, the prompt appears in indices
        # [:len(prompt_ids) - 1] in the label sequence (not [:len(prompt_ids)]).
        if prompt_len > 1:
            chosen_loss_mask[:prompt_len - 1] = 0
            rejected_loss_mask[:prompt_len - 1] = 0

        # After masking, we should have at least 1 unmasked answer token
        if chosen_loss_mask.sum().item() == 0 or rejected_loss_mask.sum().item() == 0:
            raise ValueError(f"Sample {idx}:{message}: No training tokens left after masking "
                         f"Prompt length: {len(prompt_ids)}, Chosen length: {len(chosen_answer_ids)}, "
                         f"Rejected length: {len(rejected_answer_ids)}.")
        
        # [2, T]
        all_input_ids = torch.stack([chosen_seq_ids, rejected_seq_ids], dim=0) # T
        all_attn_mask = torch.stack([chosen_seq_attn_mask, rejected_seq_attn_mask], dim=0)
        # [2, T-1]
        all_loss_mask = torch.stack([chosen_loss_mask, rejected_loss_mask], dim=0) # [T-1

        return {
            "input_ids": all_input_ids, # [2, T]
            "attn_mask": all_attn_mask, # [2, T]
            "loss_mask": all_loss_mask, # [2, T-1]
        }

    def _check_seq(self, message, prompt_len, total_seq_len, seq_ids, seq_attn_mask):
        
        # 5. Validate minimum sequence length
        if total_seq_len < 2:
            raise ValueError(f"{message}: Sequence too short: prompt + answer must be at least 2 tokens (got {total_seq_len})")

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

        return seq_ids, seq_attn_mask

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
                    'answer': "I'm good, thanks!",
                    'rejected_answer': "I'm bad, thanks!"
                   },
                   {'prompt': [{"role": "user", "content": "What is the meaning of life?"}],
                    'answer': "The meaning of life is 2000002.",
                    'rejected_answer': "The meaning of life is 2000003."
                   },
                   {'prompt': [{"role": "user", "content": "What is the meaning of the universe?"}],
                    'answer': "The meaning of the universe is galaxy plus 2.",
                    'rejected_answer': "The meaning of the universe is galaxy plus 3."
                   },
                   {'prompt': [{"role": "user", "content": "This is is a just rather long prompt that is going to be tokenized. This is a test to make sure the dataset works."}],
                    'answer': "This is a test to make sure the dataset works.",
                    'rejected_answer': "This is a test to make sure the dataset works."
                   },
                    {'prompt': [{"role": "system", "content": "You are a concise assistant."},
                                {"role": "user", "content": "Give me a weird two-word nickname."},
                                {"role": "assistant", "content": "Neon Pickle."},
                                {"role": "user", "content": "Now give a different one."}],
                    'answer': "Velvet Comet.",
                    'rejected_answer': "Velvet Comet."
                    },
                    {'prompt': [{"role": "user", "content": "Random fact, but fake."},
                                {"role": "assistant", "content": "Otters invented the first toast."},
                                {"role": "user", "content": "Another fake fact."}],
                    'answer': "Clouds are just shy mountains.",
                    'rejected_answer': "Clouds are just shy mountains."
                    }

                   ]
    df = pd.DataFrame(random_prompts)
    df.to_parquet("./promptonly.parquet", index=False)

    dataset = PreferenceFeed(
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
                                                  dataset_cls=PreferenceFeed,
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