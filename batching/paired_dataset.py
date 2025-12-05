import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import os

class PairedDataset(Dataset):
    '''
        This is a general dataset that works for any source data that has the following formant:
        {
            "prompt": [{"role": "system", "content": "this is a system prompt"}, {"role": "user", "content": "this is a user prompt"}],
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
        assert os.path.exists(data_path), "data does not exist"
        # add this assert yo mask sure that tokenizer has a pad token (or if not, we already added during loading)
        assert tokenizer.pad_token_id is not None, "tokenizer must have a pad token"

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
            self.data = pd.read_parquet(self.data_path)
        except Exception as e:
            raise Exception(f"Failed to load data from {self.data_path}: {str(e)}")

        self.len_data = len(self.data)

    def __getitem__(self, idx):
        '''
           data is pandas series with the following format:
           {
               "messages": [{"role": "system", "content": "this is a system prompt"}, {"role": "user", "content": "this is a user prompt"}],
               "answer": "this is an answer",
               ...
           }
           Note system prompt is optional.
        '''
        current_sample = self.data.iloc[idx]
        message = current_sample[self.prompt_key]
        answer  = current_sample[self.answer_key]

        # now tokenize the prompt
        prompt_chat_str = self.tokenizer.apply_chat_template(messages=message, add_generation_prompt=True, tokenize=False)
        prompt_ids_output = self.tokenizer(prompt_chat_str, return_tensors='pt', add_special_tokens=False)
        prompt_ids = prompt_ids_output['input_ids'][0]
        prompt_attn_mask = prompt_ids_output['attention_mask'][0]

        # label/answer
        answer_chat_str = answer + self.tokenizer.decode(self.tokenizer.eos_token_id)
        answer_ids_output = self.tokenizer(answer_chat_str, return_tensors='pt', add_special_tokens=False)
        answer_ids = answer_ids_output['input_ids'][0]
        answer_attn_mask = answer_ids_output['attention_mask'][0]

        total_seq_len_just_ids = len(prompt_ids) + len(answer_ids)
        # if data is text, it should be 1-dim. adding dim=-1 for future compatibility.
        seq_ids = torch.cat((prompt_ids, answer_ids), dim=-1)
        seq_attn_mask = torch.cat((prompt_attn_mask, answer_attn_mask), dim=-1)

        # length check
        if total_seq_len_just_ids > self.max_seq_len:
            # this should be ideally handled in data-preprocessing step
            raise ValueError(f"Total length of prompt and answer is {total_seq_len_just_ids}, which is greater than max_seq_len {self.max_seq_len}")

        elif total_seq_len_just_ids < self.max_seq_len:
            # pad the sequence
            padding_len = self.max_seq_len - total_seq_len_just_ids
            # add padding tokens to ids
            padding_tokens = torch.ones(size=(padding_len,), dtype=seq_ids.dtype) * self.tokenizer.pad_token_id
            seq_ids = torch.cat((seq_ids, padding_tokens), dim=-1)

            # add zeros to attention mask as padding
            padding_attn_mask = torch.zeros(size=(padding_len,), dtype=seq_attn_mask.dtype)
            seq_attn_mask = torch.cat((seq_attn_mask, padding_attn_mask), dim=-1)

        # loss mask
        # We add eos to end of each seq, so we need to make sure we do not include it in loss.
        # also we don't include prompt in the loss calculation. so we need to account for that too.
        # so we need to make a loss mask that is 1 for all tokens except the last token of each seq.
        loss_mask = seq_attn_mask.clone()
        # prompt tokens are not included in loss
        loss_mask[:len(prompt_ids)] = 0

        # eos token is not included in loss but inorder to that we need to consider min(total_seq_len_just_ids, max_seq_len - 1)
        # because we might have already padded the sequence.
        loss_mask[min(total_seq_len_just_ids, self.max_seq_len - 1)] = 0

        return {
            "seq_ids": seq_ids,
            "seq_attn_mask": seq_attn_mask,
            "loss_mask": loss_mask,
        }

    def __len__(self):
        return self.len_data