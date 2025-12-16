import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import os
from datasets import load_dataset

class PairedDataset(Dataset):
    '''
        This is a general dataset that works for any source data that has the following formant:
        {
            "prompt": [{"role": "system", "content": "this is a system prompt"},
                       {"role": "user", "content": "this is a user prompt"}],
            "answer": "this is an answer",
        }
        The data should be in a parquet format.
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

    def __getitem__(self, idx):
        '''
           data is a dict with the following format:
           {
               "messages": [{"role": "system", "content": "this is a system prompt"},
                            {"role": "user", "content": "this is a user prompt"}],
               "answer": "this is an answer",
               ...
           }
           Note system prompt is optional.
        '''
        current_sample = self.data[idx]
        message = current_sample[self.prompt_key]
        answer  = current_sample[self.answer_key]

        # answer cannot be empty
        if not answer or (isinstance(answer, str) and answer.strip() == ""):
            raise ValueError(f"Sample {current_sample}: Answer cannot be empty or whitespace-only")

        # message cannot be empty
        if not message or (isinstance(message, list) and len(message) == 0):
            raise ValueError(f"Sample {idx}:{current_sample}: Prompt/message cannot be empty")

        # 1. Tokenize the prompt
        # When tokenize=True and return_tensors='pt', it returns shape [1, seq_len]
        # [0]: [1, seq_len] -> [seq_len]
        prompt_ids = self.tokenizer.apply_chat_template(
                                                        conversation=message,
                                                        add_generation_prompt=True,
                                                        tokenize=True,
                                                        return_tensors='pt'
                                                        )[0]
        prompt_attn_mask = torch.ones_like(prompt_ids)
        prompt_len = len(prompt_ids)

        # 2. Validate prompt length
        if prompt_len >= self.max_seq_len or prompt_len == 0:
            raise ValueError(f"Prompt in sample {idx}:{current_sample}: too long or empty: "
                             f"prompt must be at most {self.max_seq_len} tokens (got {prompt_len})")

        # 3. Tokenize answer + add EOS
        answer_ids_output = self.tokenizer(answer,
                                           return_tensors='pt',
                                           add_special_tokens=False)
        # Append eos token id manually
        # this is usefult to cover cases where the answer doesn't end with a space,
        # the tokenizer might merge the last word and the EOS token into a
        # single unknown or different token.
        eos_tensor = torch.tensor([self.tokenizer.eos_token_id],
                                   dtype=answer_ids_output['input_ids'].dtype)
        answer_ids = torch.cat([answer_ids_output['input_ids'][0], eos_tensor], dim=0)
        answer_attn_mask = torch.cat([answer_ids_output['attention_mask'][0],
                                      torch.tensor([1], dtype=answer_ids_output['attention_mask'].dtype)], dim=0)
        # Validate answer has at least one token besides EOS
        if len(answer_ids) <= 1:
            raise ValueError(
                f"Sample {idx}:{current_sample}: Answer must tokenize to at least one token (excluding EOS). "
                f"Got {len(answer_ids)} tokens total."
            )

        seq_ids = torch.cat((prompt_ids, answer_ids), dim=-1).to(dtype=torch.long)
        seq_attn_mask = torch.cat((prompt_attn_mask, answer_attn_mask), dim=-1)
        total_seq_len = len(seq_ids)

        # 4. Validate minimum sequence length
        if total_seq_len < 2:
            raise ValueError(f"Sequence too short: prompt + answer must be at least 2 tokens (got {total_seq_len})")

        # 5. length check
        if total_seq_len > self.max_seq_len:
            # this should be ideally handled in data-preprocessing step
            # we might lose the EOS token here. This is acceptable in SFT training though
            # as the model learns max length reached.
            seq_ids = seq_ids[:self.max_seq_len]
            seq_attn_mask = seq_attn_mask[:self.max_seq_len]
            total_seq_len = len(seq_ids)

        # 6. pad if necessary
        elif total_seq_len < self.max_seq_len:
            padding_len = self.max_seq_len - total_seq_len

            # add padding tokens to ids
            padding_tokens = torch.full((padding_len,), self.tokenizer.pad_token_id, dtype=seq_ids.dtype)
            seq_ids = torch.cat((seq_ids, padding_tokens), dim=-1)

            # add zeros to attention mask as padding
            padding_attn_mask = torch.zeros(size=(padding_len,), dtype=seq_attn_mask.dtype)
            seq_attn_mask = torch.cat((seq_attn_mask, padding_attn_mask), dim=-1)

        # Loss mask:
        # Labels are created by shifting seq_ids by one position, so they have length T-1.
        # Therefore, the loss mask must also be of shape [T-1].
        # Padding is already handled by seq_attn_mask if any (pads are zero),
        # so no extra padding logic is needed.
        loss_mask = seq_attn_mask[1:].clone()

        # Mask out prompt tokens.
        # Since labels are shifted by one position, the prompt appears in indices
        # [:len(prompt_ids) - 1] in the label sequence (not [:len(prompt_ids)]).
        if prompt_len > 1:
            loss_mask[:prompt_len - 1] = 0

        # After masking, we should have at least 1 unmasked answer token
        if loss_mask.sum().item() == 0:
            raise ValueError(f"Sample {idx}:{current_sample[self.prompt_key]}: No training tokens left after masking "
                         f"Prompt length: {len(prompt_ids)}, Answer length: {len(answer_ids)}, "
                         f"Total length: {total_seq_len}.")

        return {
            "seq_ids": seq_ids, # T
            "seq_attn_mask": seq_attn_mask, # T
            "loss_mask": loss_mask, # T-1
        }

    def __len__(self):
        return self.len_data

if __name__ == "__main__":
