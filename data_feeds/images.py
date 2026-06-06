import os
from torch.utils.data import Dataset
from datasets import load_dataset
from data_feeds.image_utils import _load_pil_image


class ImagePromptsFeed(Dataset):
    '''
        Returns text prompts plus multi_modal_data payloads for vLLM image rollouts.
    '''
    def __init__(self,
                prompt_key: str,
                tokenizer,
                max_seq_len: int,
                data_path: str,
                solution_key: str = None,
                image_key: str = "image_bytes",
                processor=None,
                model_adapter=None,
                ):
        assert prompt_key != "", "prompt_key cannot be empty"
        assert max_seq_len > 0, "max_seq_len must be > 0"
        assert tokenizer is not None, "tokenizer cannot be None"
        assert processor is not None, "processor cannot be None"
        assert isinstance(data_path, str), "data_path must be a string"
        assert os.path.exists(os.path.expanduser(data_path)), f"{data_path} does not exist"
        assert image_key != "", "image_key cannot be empty"

        self.prompt_key = prompt_key
        self.solution_key = solution_key or None
        self.max_seq_len = int(max_seq_len)
        self.tokenizer = tokenizer
        self.processor = processor
        self.data_path = data_path
        self.image_key = image_key
        self.model_adapter = model_adapter
        self._load_data()

    def _load_data(self):
        try:
            self.data = load_dataset("parquet", data_files=self.data_path, split="train")
        except Exception as e:
            raise Exception(f"Failed to load data from {self.data_path}: {str(e)}")

        self.len_data = len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.prompt_key not in sample:
            raise KeyError(f"Missing key '{self.prompt_key}' in sample {sample}: keys={list(sample.keys())}")
        if self.image_key not in sample:
            raise KeyError(f"Missing key '{self.image_key}' in sample keys={list(sample.keys())}")

        message = sample[self.prompt_key]
        if not message or (isinstance(message, list) and len(message) == 0):
            raise ValueError(f"Sample {idx}:{sample}: Prompt cannot be empty")

        if self.model_adapter is not None and hasattr(self.model_adapter, "prepare_messages"):
            message = self.model_adapter.prepare_messages(message)

        prompt_text = self.processor.apply_chat_template(
            conversation=message,
            add_generation_prompt=True,
            tokenize=False,
            return_tensors=None,
            skip_special_tokens=False,
        )
        pil = _load_pil_image(sample[self.image_key])
        out = {"prompt": prompt_text, "text": prompt_text, "multi_modal_data": {"image": pil}}
        if self.solution_key:
            out["solution"] = sample[self.solution_key]
        return out

    def __len__(self):
        return self.len_data

    def collate_fn(self, batch):
        return batch
