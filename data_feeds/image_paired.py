import torch

from data_feeds.paired import PairedFeed
from data_feeds.image_utils import load_images


class ImagePairedFeed(PairedFeed):
    '''
        VLM variant of PairedFeed that inherits PairedFeed's answer encoding, padding, and single/multi-turn loss
        masking UNCHANGED (the masking depends only on token positions, and the expanded
        image tokens fall inside the masked prompt span). It overrides the prompt encoding
        (_encode_prompt) to tokenize via a processor and returns the multimodal
        tensors (e.g. pixel_values, image_grid_thw).

        Multimodal tensors are handled in two generic kinds:
          - image "bag" tensors (pixel_values, image_grid_thw, ...): variable per sample,
            concatenated along dim 0 in collate.
          - per-token tensors aligned to the sequence (e.g. gemma-3 token_type_ids): the
            processor returns them only for the prompt, so _align_mm extends them to the full
            padded sequence (image positions kept, answer+pad = 0) and collate stacks them [B,T].

        Coverage: HF VLMs whose extra inputs are image-bag and/or per-token are supported
        (Qwen2.5-VL, Gemma-3/PaliGemma, LLaVA, InternVL). Models with bespoke 4-D inputs
        (e.g. Mllama cross_attention_mask) are not handled by this generic path for now.
    '''
    # Per-token multimodal keys aligned to the sequence: _align_mm pads them to the full
    # sequence (fill 0 = non-special/text) and collate stacks them [B, T]. Add a model's
    # per-token key here to support it (token_type_ids covers Gemma-3 / PaliGemma).
    _SEQ_ALIGNED_MM_KEYS = ("token_type_ids",)

    def __init__(self,
                 prompt_key,
                 answer_key,
                 max_seq_len,
                 processor=None,
                 image_key=None,
                 max_image_pixels=None,
                 data_path=""):
        assert processor is not None, "ImagePairedFeed requires a processor (AutoProcessor)"
        assert image_key, "ImagePairedFeed requires image_key"
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
            validation/dispatch/masking; the overridden _encode_prompt reads self._cur_images.
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

            Args:
                multimodal_data: dict[str, torch.Tensor] from _encode_prompt. Per-token keys
                                 (e.g. token_type_ids) are [1, prompt_len] int tensors;
                                 image-bag keys (e.g. pixel_values) are left untouched.
                prompt_len:      int, number of prompt tokens.
                seq_len:         int, full padded sequence length (== max_seq_len).
            Returns:
                dict[str, torch.Tensor] with each _SEQ_ALIGNED_MM_KEYS tensor replaced by a
                1-D [seq_len] tensor; all other keys unchanged.
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
            # copy prompt values into a zero vector of the full length; 
            # answer+pad stay 0, e.g., v=[0,1,1,1,0] (prompt_len=5), 
            # seq_len=8 -> [0,1,1,1,0,0,0,0]
            full = torch.zeros(seq_len, dtype=v.dtype)
            n = min(prompt_len, seq_len, v.shape[0])
            full[:n] = v[:n]
            out[key] = full
        return out

    def _inject_image_placeholders(self, message, num_images):
        '''
            Inject num_images {"type": "image"} placeholders into the FIRST user turn and
            return (new_message, n_placeholders_placed). Other turns pass through unchanged.
            For an incremental prefix that does not yet include a user turn, nothing is
            placed (n=0), so the matching number of images is passed to the processor.

            Example:
                message  = [{"role": "user", "content": "What is in this image?"}]
                n_images = 2

                rewrites the user turn content to structured form:
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
            Encode a (sub)conversation via the processor, returning (prompt_ids[1D], mm_dict).
            mm_dict holds every tensor the processor returned except input_ids/attention_mask
            (e.g. pixel_values, image_grid_thw, and per-token keys like token_type_ids); it is
            empty when the processor returns nothing extra (typical text-only encoding).
        '''
        # Processor outputs the base path already owns, so they are not
        # carried as mm tensors.
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

    def collate_fn(self, batch):
        '''
            Collate a batch of per-sample dicts. Stacked into [B, T]: the fixed-length text
            tensors AND any per-token (sequence-aligned) multimodal tensor in
            _SEQ_ALIGNED_MM_KEYS (e.g. token_type_ids). Concatenated along dim 0: the
            variable-shape image 'bag' (pixel_values/image_grid_thw), which is how HF VLMs
            consume them (a flat stack of images/patches whose per-image split is described by
            the grid). Samples without images simply contribute nothing to the image keys.

            Qwen2.5-VL example (T=128; A: 224x224 image, B: 336x336 image, C: text-only).
            patches per image = grid_t * grid_h * grid_w, where grid_h/grid_w = side/patch_size
            and patch_size=14 (the ViT patch size):
                A: 224/14=16 -> 1*16*16 = 256 patches
                B: 336/14=24 -> 1*24*24 = 576 patches
            1176 = channels * temporal * patch * patch = 3*2*14*14 (flattened per-patch dim).
            All numbers are Qwen2.5-VL-specific.
                batch = [# A
                         {"input_ids": [128], "attn_mask": [128], "loss_mask": [127],
                         "pixel_values": [256, 1176], "image_grid_thw": [1, 3]},
                         # B
                         {"input_ids": [128], "attn_mask": [128], "loss_mask": [127],
                         "pixel_values": [576, 1176], "image_grid_thw": [1, 3]},
                         # C (no image keys)
                         {"input_ids": [128], "attn_mask": [128], "loss_mask": [127]},
                         ]

                returns: {"input_ids":      [3, 128],     # stack
                          "attn_mask":      [3, 128],     # stack
                          "loss_mask":      [3, 127],     # stack
                          "pixel_values":   [832, 1176],  # cat dim 0: 256 + 576 (C contributes nothing)
                          "image_grid_thw": [2, 3]}       # cat dim 0: A and B only
            The result is directly model(**batch)-ready: image_grid_thw tells the model how
            to slice the 832 pixel_values rows back into the 2 images.
        '''
        # Stack the fixed-length text tensors and per-token (sequence-aligned) mm tensors.
        # A sequence-aligned key (e.g. token_type_ids) may be absent on a text-only sample in
        # an otherwise-image batch; fill the missing entries with zeros (all-text) so the
        # batch dim stays aligned with input_ids. Keys absent from every sample are skipped.
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