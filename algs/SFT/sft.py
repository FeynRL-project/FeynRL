import torch

class SFT:
    def __init__(self, model_engine, optimizer, normalize_loss=False, world_size=1):

        self.model_engine = model_engine
        self.optimizer = optimizer
        self.normalize_loss = normalize_loss
        self.world_size = world_size

        # use cross entropy loss
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    def compute_loss(self, logits, target_ids, loss_mask, ga_denom=None, ga_steps=1, is_training=False):
        '''
         This implements sum_{i=1}^{N} log p(y_i|x_i)
         target_ids is [B, T -1]
         logits is [B, T -1, vocab_size]
         ga_denom: total valid tokens across ALL GPUs in the GA window.
         ga_steps: gradient accumulation steps.
         is_training: when True, scale loss to match the effective-batch objective.
        '''
        # [B, T -1, vocab_size]
        _, _, vocab_size = logits.shape

        # flatten logits across batch and seq_len before computing loss
        # so logits is [B * (T -1), vocab_size]
        # Casting logits to float32 to avoid overflow on large vocabs
        logits = logits.view(-1, vocab_size).float()
        # flatten y as well:  [B, T -1] -->  [B * (T -1)]
        target_ids = target_ids.view(-1)

        # per token loss
        per_token_loss = self.loss_fn(logits, target_ids)

        # Apply mask to loss to remove any token that should not be considered in loss (e.g., padding tokens)
        loss_mask = loss_mask.view(-1).to(dtype=per_token_loss.dtype)  # [B * (T - 1)]
        masked_per_token_loss = per_token_loss * loss_mask

        loss_sum = masked_per_token_loss.sum()
        num_tokens = float(loss_mask.sum().item())

        loss = loss_sum
        if is_training:
            if ga_steps < 1:
                raise ValueError(f"ga_steps must be >= 1, got {ga_steps}")

            # DeepSpeed averages gradients across data-parallel ranks and scales them
            # with respect to gradient accumulation. Hence we multiply by ga_steps to cancel out
            # the averaging effect of deepspeed.
            dp_scale = ga_steps * self.world_size

            if self.normalize_loss:
                if ga_denom is None or ga_denom <= 0:
                    raise ValueError(f"normalize_loss=True requires positive ga_denom for backward, got {ga_denom}")
                # True global mean over supervised tokens in the full effective batch.
                loss = loss_sum * (dp_scale / ga_denom)

            else:
                # True global sum over supervised tokens in the full effective batch.
                loss = loss_sum * dp_scale

        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError(f"SFT loss is NaN or Inf: {loss.item()}")

        return loss, loss_sum.item(), num_tokens

    def forward(self, batch):
        '''
            batch['input_ids/attn_mask'] are [B, T]
            batch['position_ids'] is [B, T] or None
            For multimodal batches, batch also carries the processor's image tensors
            (e.g. pixel_values, image_grid_thw); they are passed straight through to the model.
            Returns:
                logits is [B, T-1, vocab_size]
                y is [B, T-1]
                loss_mask is [B, T-1]
        '''
        # input_ids and att_mask are [B, T]
        input_ids = batch['input_ids']
        att_mask  = batch['attn_mask']
        # loss_mask is [B, T - 1]
        loss_mask = batch['loss_mask']

        # if pos_ids is not provided, hf will add it automatically. For vlm models this must
        # stay None so the model computes its own multimodal position ids (e.g. Qwen mRoPE).
        pos_ids = batch.get('position_ids', None)
        if pos_ids is not None:
            pos_ids = pos_ids.to(att_mask.device)

        # Forward any multimodal tensors the data feed added.
        # Text-only batches carry none, so this is a no-op for llm.
        reserved_keys = ('input_ids', 'attn_mask', 'loss_mask', 'position_ids')
        mm_kwargs = {k: v for k, v in batch.items() if k not in reserved_keys}

        # Processor image tensors (e.g. pixel_values) come in float32; cast floating mm tensors
        # to the model's compute dtype (e.g. bf16) so the vision tower doesn't hit a dtype
        # mismatch. Integer tensors (image_grid_thw, token_type_ids) are left unchanged. We use
        # the input-embedding dtype, which stays the base compute dtype even under PEFT/LoRA
        # (where adapter params may be float32).
        if mm_kwargs:
            compute_dtype = self.model_engine.module.get_input_embeddings().weight.dtype
            mm_kwargs = {k: (v.to(compute_dtype) if torch.is_floating_point(v) else v)
                         for k, v in mm_kwargs.items()}

        # feed data to model
        output = self.model_engine(input_ids=input_ids,
                                   attention_mask=att_mask,
                                   position_ids=pos_ids,
                                   use_cache=False,
                                   **mm_kwargs)

        # [B, T, vocab_size]
        every_token_logits = output.logits
        # remember we use token t to predict token t+1, hence no need to predict last
        # token's output (e.g., <eos>) and we remove it from logits.
        logits = every_token_logits[:, :-1, :].contiguous()

        # target_ids would be input_ids shifted by one
        # so the size is [B, T-1]
        target_ids = input_ids[:, 1:].contiguous()

        return logits, target_ids, loss_mask

    def eval_step(self, micro_batch):
        '''
           This implements a single validation step per rank/gpu.
           Setting model to eval mode and torch.no_grad() context are done in main.
        '''
        # forward pass per gpu/rank
        logits, target_ids, loss_mask = self.forward(micro_batch)
        # compute loss pass
        _, loss_sum, num_tokens = self.compute_loss(logits=logits, target_ids=target_ids, loss_mask=loss_mask)

        return {"loss_sum": float(loss_sum), "num_tokens": float(num_tokens)}

    def train_step(self, micro_batch, ga_denom=None, ga_steps=1):
        '''
           This implements a single training step per rank/gpu
           for given micro_batch_size_per_gpu.
           ga_denom: total valid tokens across all GPUs in the GA window.
           ga_steps: gradient accumulation steps.
        '''
        # Don't need to zero_grad() here as ds handles gradient zeroing
        # internally after step() when gradient_accumulation_steps boundary is reached.
        # 1. forward pass per gpu/rank
        logits, target_ids, loss_mask = self.forward(micro_batch)

        # 2. compute loss pass
        loss, loss_sum, num_tokens = self.compute_loss(logits=logits,
                                                       target_ids=target_ids,
                                                       loss_mask=loss_mask,
                                                       ga_denom=ga_denom,
                                                       ga_steps=ga_steps,
                                                       is_training=True)

        # 3. backward step
        # deepspeed aggregates gradients and only updates weights when accumulation_steps is reached.
        self.model_engine.backward(loss)

        # 4. optimizer step
        self.model_engine.step()

        return {"loss": float(loss.item()), "loss_sum": float(loss_sum), "num_tokens": float(num_tokens)}
