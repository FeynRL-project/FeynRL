import os

from data_feeds.prompts import PromptsFeed
from data_feeds.image_utils import load_images

class ImagePromptsFeed(PromptsFeed):
    '''
        Vision-language variant of PromptsFeed for the vLLM rollout/eval path.
        Unlike the SFT feed (ImagePairedFeed), this feed does NOT pre-expand
        the image placeholder into the processor's image tokens. vLLM expands the
        placeholder itself from multi_modal_data; if we ALSO pre-expand, the placeholder
        run gets expanded twice, corrupting the image conditioning. So we defer expansion to vLLM.

        Each item carries:
          - prompt           : the chat-template TEXT with one image placeholder per image
                               (single, un-expanded). vLLM tokenizes + expands it once.
          - multi_modal_data : {"image": [PIL images]} consumed by vLLM's processor.
        The raw PIL images are passed through; vLLM re-processes them, so the data feed and
        the engine must use the same model/processor (they do -- same model path).
    '''
    def __init__(self,
                 prompt_key,
                 max_seq_len,
                 data_path,
                 processor=None,
                 image_key=None,
                 max_image_pixels=None,
                 solution_key=None):
        assert processor is not None, "ImagePromptsFeed requires a processor (AutoProcessor)"
        assert image_key, "ImagePromptsFeed requires image_key"
        # The base validates/loads via self.tokenizer. For a VLM this is the processor's tokenizer.
        super().__init__(prompt_key=prompt_key,
                         tokenizer=processor.tokenizer,
                         max_seq_len=max_seq_len,
                         data_path=data_path,
                         solution_key=solution_key)
        self.processor = processor
        self.image_key = image_key
        self.max_image_pixels = max_image_pixels

    def _inject_image_placeholders(self, message, num_images):
        '''
            Inject num_images {"type": "image"} placeholders into the FIRST user turn (same as
            ImagePairedFeed). Returns (new_message, n_placeholders_placed).
        '''
        if num_images == 0:
            return message, 0

        new_message = []
        placed = 0
        for turn in message:
            if placed == 0 and turn.get("role") == "user":
                content = turn.get("content", "")
                mm_content = [{"type": "image"} for _ in range(num_images)]
                if isinstance(content, list):
                    mm_content.extend(content)
                else:
                    mm_content.append({"type": "text", "text": content})
                new_message.append({**turn, "content": mm_content})
                placed = num_images
            else:
                new_message.append(turn)

        return new_message, placed

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.prompt_key not in sample:
            raise KeyError(f"Missing key '{self.prompt_key}' in sample {sample}: keys={list(sample.keys())}")

        message = sample[self.prompt_key]
        if not message or (isinstance(message, list) and len(message) == 0):
            raise ValueError(f"Sample {idx}:{sample}: Prompt cannot be empty")

        images = load_images(sample.get(self.image_key), self.max_image_pixels)
        mm_message, n_used = self._inject_image_placeholders(message, len(images))

        # Render the prompt as TEXT with one (un-expanded) image placeholder per image.
        text = self.processor.apply_chat_template(mm_message,
                                                  tokenize=False,
                                                  add_generation_prompt=True)

        # Length guard: compute the HF-expanded prompt length (processor WITH images)
        # only to validate against max_seq_len. These expanded ids are NOT sent to vLLM
        # We send the text prompt and let vLLM expand exactly once. This keeps the
        # guard accurate (counts the real image tokens) without double-expanding.
        proc_kwargs = {"text": text, "return_tensors": "pt"}
        if n_used > 0:
            proc_kwargs["images"] = images[:n_used]
        expanded_len = self.processor(**proc_kwargs)["input_ids"].shape[1]

        if expanded_len == 0:
            raise ValueError(f"Sample {idx}:{sample}: tokenization produced an empty prompt")

        if expanded_len >= self.max_seq_len:
            raise ValueError(f"Prompt in sample {idx}:{sample}: too long: prompt (incl. image tokens) "
                             f"must be at most {self.max_seq_len} tokens (got {expanded_len})")

        # vllm TextPrompt: the text (single placeholder) + raw images. vLLM expands once.
        out = {"prompt": text}
        if n_used > 0:
            out["multi_modal_data"] = {"image": images[:n_used]}
        if self.solution_key:
            out["solution"] = sample[self.solution_key]
        return out