import torch

from data_feeds.preference import PreferenceFeed
from data_feeds.image_utils import load_images


class ImagePreferenceFeed(PreferenceFeed):
    '''
        VLM variant of PreferenceFeed for image+text DPO. Mirrors ImagePairedFeed: it inherits
        PreferenceFeed's chosen/rejected encoding, padding, and loss masking UNCHANGED, and
        overrides the prompt encoding (_encode_prompt) to tokenize via a processor with the
        sample's images (which expands the image placeholder into image tokens and returns the
        multimodal tensors). The masking is untouched because it depends only on token
        positions, and the expanded image tokens fall inside the masked prompt span.

        Chosen and rejected share the same prompt and the same image(s). _pair_mm aligns the
        (shared, prompt-only) multimodal tensors to the paired [2, T] structure:
          - per-token tensors (token_type_ids) are identical for chosen/rejected -> stacked [2, T];
          - image-bag tensors (pixel_values, image_grid_thw) are duplicated for chosen+rejected
            so collate's cat-in-sample-order yields a bag aligned with the DPO forward's
            [B, 2, T] -> [2B, T] flatten ([chosen0, rejected0, chosen1, rejected1, ...]).

        Coverage matches ImagePairedFeed (Qwen2.5-VL, Gemma-3/PaliGemma, LLaVA, InternVL).
    '''
    # Per-token multimodal keys aligned to the sequence (see ImagePairedFeed). _align_mm pads
    # them to the full sequence and collate/_pair_mm stack them. token_type_ids covers Gemma-3.
    _SEQ_ALIGNED_MM_KEYS = ("token_type_ids",)

    def __init__(self,
                 prompt_key,
                 answer_key,
                 max_seq_len,
                 processor=None,
                 image_key=None,
                 max_image_pixels=None,
                 data_path=""):
        assert processor is not None, "ImagePreferenceFeed requires a processor (AutoProcessor)"
        assert image_key, "ImagePreferenceFeed requires image_key"
        # The base class drives text tokenization through self.tokenizer and for a VLM that
        # is the processor's tokenizer (pad/eos already ensured upstream in main).
        super().__init__(prompt_key=prompt_key,
                         answer_key=answer_key,
                         max_seq_len=max_seq_len,
                         tokenizer=processor.tokenizer,
                         data_path=data_path)
        self.processor = processor
        self.image_key = image_key
        self.max_image_pixels = max_image_pixels
        # Per-sample scratch which is set in __getitem__ and read by _encode_prompt.
        self._cur_images = []

    def __getitem__(self, idx):
        '''
            Decode this sample's images once, then let the base __getitem__ run the normal
            validation/masking; the overridden _encode_prompt reads self._cur_images.
        '''
        idx = int(idx)
        self._cur_images = load_images(self.data[idx].get(self.image_key), self.max_image_pixels)
        return super().__getitem__(idx)

    def _align_mm(self, multimodal_data, prompt_len, seq_len):
        '''
            Extend per-token multimodal tensors in _SEQ_ALIGNED_MM_KEYS (which the processor
            produced only for the prompt, shape [1, prompt_len]) to the full padded sequence
            [seq_len]: prompt positions are kept, the answer+pad region is filled with 0
            (non-special / text). Image-bag tensors are left unchanged.
        '''
        if not multimodal_data:
            return multimodal_data

        out = dict(multimodal_data)
        for key in self._SEQ_ALIGNED_MM_KEYS:
            if key not in out:
                continue
            v = out[key]
            # [1, prompt_len] -> [prompt_len]
            if v.dim() == 2:
                v = v[0]
            # copy prompt values into a zero vector of the full length; answer+pad stay 0,
            # e.g., v=[0,1,1,1,0] (prompt_len=5), seq_len=8 -> [0,1,1,1,0,0,0,0]
            full = torch.zeros(seq_len, dtype=v.dtype)
            n = min(prompt_len, seq_len, v.shape[0])
            full[:n] = v[:n]
            out[key] = full
        return out

    def _inject_image_placeholders(self, message, num_images):
        '''
            Inject num_images {"type": "image"} placeholders into the FIRST user turn and
            return (new_message, n_placeholders_placed). Other turns pass through unchanged.

            Example:
                message  = [{"role": "user", "content": "What is in this image?"}]
                n_images = 2
                -> user turn content becomes
                [{"type": "image"}, {"type": "image"},
                 {"type": "text", "text": "What is in this image?"}]
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

    def _encode_prompt(self, message, add_generation_prompt):
        '''
            Encode the (shared) prompt via the processor, returning (prompt_ids[1D], mm_dict).
            mm_dict holds every tensor the processor returned except input_ids/attention_mask
            (e.g. pixel_values, image_grid_thw, token_type_ids); empty for a text-only encoding.
        '''
        # Processor outputs the base path already owns, so they are not carried as mm tensors.
        skip_keys = ("input_ids", "attention_mask")

        images = self._cur_images or []
        mm_message, n_used = self._inject_image_placeholders(message, len(images))

        text = self.processor.apply_chat_template(mm_message,
                                                  tokenize=False,
                                                  add_generation_prompt=add_generation_prompt)

        proc_kwargs = {"text": text, "return_tensors": "pt"}
        if n_used > 0:
            proc_kwargs["images"] = images[:n_used]

        enc = self.processor(**proc_kwargs)
        prompt_ids = enc["input_ids"][0]
        mm = {k: v for k, v in enc.items() if k not in skip_keys}
        return prompt_ids, mm

    def _pair_mm(self, multimodal_data, prompt_len, seq_len):
        '''
            Align the shared, prompt-only multimodal tensors to the paired [2, T]
            chosen/rejected structure. Empty for text-only (multimodal_data == {}).
              - per-token tensors -> stacked [2, seq_len] (chosen and rejected identical);
              - image-bag tensors -> duplicated cat([v, v], dim=0) for chosen + rejected.
        '''
        if not multimodal_data:
            return {}

        aligned = self._align_mm(multimodal_data, prompt_len, seq_len)
        out = {}
        for key, v in aligned.items():
            if key in self._SEQ_ALIGNED_MM_KEYS:
                out[key] = torch.stack([v, v], dim=0)   # [seq_len] -> [2, seq_len]
            else:
                out[key] = torch.cat([v, v], dim=0)      # image bag for chosen + rejected
        return out

    def collate_fn(self, batch):
        '''
            Collate a batch of per-sample dicts. Stacked into [B, 2, T]: the paired text tensors
            AND any per-token (sequence-aligned) multimodal tensor in _SEQ_ALIGNED_MM_KEYS.
            Concatenated along dim 0: the variable-shape image 'bag' (pixel_values/
            image_grid_thw), already duplicated per sample by _pair_mm so the cat-in-sample-order
            aligns with the DPO forward's [B, 2, T] -> [2B, T] flatten. Text-only samples
            contribute nothing to the image keys.
        '''
        # Stack the paired text tensors and per-token (sequence-aligned) mm tensors.
        # A sequence-aligned key may be absent on a text-only sample in an otherwise-image
        # batch; fill the missing entries with zeros so the batch dim stays aligned.
        stack_keys = ("input_ids", "attn_mask", "loss_mask") + self._SEQ_ALIGNED_MM_KEYS
        out = {}
        for k in stack_keys:
            present = [ex.get(k) for ex in batch]
            ref = next((v for v in present if v is not None), None)
            if ref is None:
                continue
            filled = [v if v is not None else torch.zeros_like(ref) for v in present]
            out[k] = torch.stack(filled, dim=0)

        # Concatenate the remaining (image-bag) tensors along dim 0.
        seen = set(stack_keys)
        for ex in batch:
            for k in ex:
                if k in seen:
                    continue
                seen.add(k)
                vals = [e[k] for e in batch if k in e]
                if vals:
                    out[k] = torch.cat(vals, dim=0)

        return out