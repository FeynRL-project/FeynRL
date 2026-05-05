import os
from torch.utils.data import Dataset
from datasets import load_dataset

class PromptsFeed(Dataset):
    '''
        Returns tokenized prompt ids as a list[int] which can be a variable length.
    '''
    def __init__(self,
                prompt_key: str,
                tokenizer,
                max_seq_len: int,
                data_path: str,
                solution_key: str = None,
                enable_thinking: bool = None,
                ):
        assert prompt_key != "", "prompt_key cannot be empty"
        assert max_seq_len > 0, "max_seq_len must be > 0"
        assert tokenizer is not None, "tokenizer cannot be None"
        assert isinstance(data_path, str), "data_path must be a string"
        assert os.path.exists(os.path.expanduser(data_path)), f"{data_path} does not exist"
        assert tokenizer.pad_token_id is not None, "tokenizer must have a pad token"
        assert tokenizer.eos_token_id is not None, "tokenizer must have an eos token"

        self.prompt_key  = prompt_key

        # this is required for reward function of datatset where solution is provided.
        if solution_key:
            self.solution_key = solution_key

        else:
            self.solution_key = None

        self.max_seq_len      = int(max_seq_len)
        self.tokenizer        = tokenizer
        self.data_path        = data_path
        self.enable_thinking  = enable_thinking
        self._load_data()

    def _load_data(self):
        '''
           Loads the data from a parquet file.
        '''
        try:
            # Loads lazily (disk/cache). split='train' is a HF datasets arg:
            #  without it we get a DatasetDict; with it we get a Dataset.
            self.data = load_dataset("parquet", data_files=self.data_path, split="train")

        except Exception as e:
            raise Exception(f"Failed to load data from {self.data_path}: {str(e)}")

        self.len_data = len(self.data)

    def __getitem__(self, idx):
        '''
           data is a dict with the following format:
           {
                "prompt": [{"role": "system", "content": "..."},
                           {"role": "user", "content": "..."}
                          ],
                "solution": "..." # optional
           }
           Note system prompt is optional.
        '''
        sample = self.data[idx]
        if self.prompt_key not in sample:
            raise KeyError(f"Missing key '{self.prompt_key}' in sample {sample}: keys={list(sample.keys())}")

        message = sample[self.prompt_key]
        # message cannot be empty
        if not message or (isinstance(message, list) and len(message) == 0):
            raise ValueError(f"Sample {idx}:{sample}: Prompt cannot be empty")

        thinking_kwargs = (
            {"enable_thinking": self.enable_thinking}
            if self.enable_thinking is not None
            else {}
        )

        # Tokenize prompt for vLLM rollout
        prompt_ids = self.tokenizer.apply_chat_template(
                                        conversation=message,
                                        add_generation_prompt=True,
                                        tokenize=True,
                                        return_tensors=None,
                                        **thinking_kwargs,
                                        )
        if not isinstance(prompt_ids, list) or len(prompt_ids) == 0:
            raise ValueError(f"Sample {idx}:{sample}: tokenization produced empty prompt_ids")

        # Validate prompt length
        if len(prompt_ids) >= self.max_seq_len:
            raise ValueError(f"Prompt in sample {idx}:{sample}: too long: "
                             f"prompt must be at most {self.max_seq_len} tokens (got {len(prompt_ids)})")

        # Get the prompt text for debugging.
        prompt_text = self.tokenizer.apply_chat_template(
                                        conversation=message,
                                        add_generation_prompt=True,
                                        tokenize=False,
                                        return_tensors=None,
                                        skip_special_tokens=False,
                                        **thinking_kwargs,
                                        )
        if self.solution_key:
            solution = sample[self.solution_key]
            return  {"prompt_token_ids": prompt_ids, "text": prompt_text, "solution": solution}

        else:
            return  {"prompt_token_ids": prompt_ids, "text": prompt_text}


    def __len__(self):
        return self.len_data

    def collate_fn(self, batch):
        '''
            Since pytorch's default collate tries to stack sequences, we need to override it.
            Otherwise, we will get all sequences must have equal size error.
            This function keeps variable-length items as python objects
        '''
        return batch

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

    random_prompts = [
        {'prompt': [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello, how are you?"}], 'solution': 'two'},
        {'prompt': [{"role": "user", "content": "What is the meaning of life?"}], 'solution': 'it is hard to say.'},
        {'prompt': [{"role": "user", "content": "What is the meaning of the universe?"}], 'solution': 'haha.'},
        {'prompt': [{"role": "user", "content": "This is is a just rather long prompt that is going to be tokenized. This is a test to make sure the dataset works."}], 'solution': '42'},
    ]
    df = pd.DataFrame(random_prompts)
    df.to_parquet("./promptonly.parquet", index=False)

    dataset = PromptsFeed(
                        prompt_key="prompt",
                        tokenizer=tokenizer,
                        max_seq_len=1024,
                        data_path="./promptonly.parquet",
                        solution_key="",
                        )
    dataloader = DataLoader(dataset,
                            batch_size=3,
                            collate_fn=dataset.collate_fn,
                            shuffle=False)
    for d in dataloader:
        print(d)
        print("\n")

    from mixed_sampler import create_prompt_dataset_and_sampler

    concat_ds, sampler, collate_fn = create_prompt_dataset_and_sampler(data_paths=["./promptonly.parquet"],
                                      prompt_key="prompt",
                                      solution_key="solution",
                                      max_seq_len=1024,
                                      tokenizer=tokenizer,
                                      train_ratios={"promptonly":1},
                                      seed=42,
                                      local_batch_size=3,
                                      dataset_cls=PromptsFeed,
                                      dynamic_ratio_every_step=False,
                                      steps_per_epoch=3,
                                      shuffle_within_batch=True,
                                      )

    dummy_loader = DataLoader(dataset=concat_ds,
                            batch_sampler=sampler,
                            num_workers=0,
                            pin_memory=True,
                            collate_fn=collate_fn,
                            )

    for d in dummy_loader:
        print(d)