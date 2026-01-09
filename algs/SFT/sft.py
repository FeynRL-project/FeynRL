import torch
import numpy as np

class SFT:
    def __init__(self,
                model_engine,
                optimizer,
                use_cache=False,
                normalize_loss=False):

        self.model_engine = model_engine
        self.optimizer = optimizer
        self.use_cache = use_cache
        self.normalize_loss = normalize_loss

        # use cross entropy loss
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    def compute_loss(self, logits, target_ids, loss_mask):
        '''
         This function implements \sum_{i=1}^{N} log p(y_i|x_i)
         target_ids is target label [B, T -1]
         logits is model prediction [B, T -1, vocab_size]
        '''
        # [B, T -1, vocab_size]
        _, _, vocab_size = logits.shape

        # flatten logits across batch and seq_len before computing loss
        # so logits is [B * (T -1), vocab_size]
        logits = logits.view(-1, vocab_size)
        # flatten y as well:  [B, T -1] -->  [B * (T -1)]
        target_ids = target_ids.view(-1)

        # per token loss
        per_token_loss = self.loss_fn(logits, target_ids)

        # We need to apply mask to loss to remove any things 
        # which should not be considered in loss (e.g., padding tokens)
        loss_mask = loss_mask.view(-1).to(dtype=per_token_loss.dtype)  # [B * (T - 1)]
        masked_per_token_loss = per_token_loss * loss_mask

        # To avoid gradient accumulation error caused by loss.mean(),
        # we use sum of loss instead but play with learning rate to account for this.
        loss = masked_per_token_loss.sum()

        # Loss_accumulated \neq Loss_full_batch when sequence lengths vary.
        # to address that, we normalize by total sequence length (constant)
        # which is fixed across gpus, not valid tokens (variable) which is loss_mask.sum().
        # This solves the gradient accumulation bug.
        if self.normalize_loss:
            total_possible_tokens = logits.shape[0]
            if total_possible_tokens == 0:
                # This shouldn't happen
                raise ValueError("Cannot compute loss: total_possible_tokens is 0")

            loss = loss / total_possible_tokens

        return loss

    def forward(self, batch):
        '''
            This function implements a single forward pass for current batch:
            batch['input_ids/attn_mask'] are [B, T]
            batch['position_ids'] is [B, T] or None
            Returns:
                logits is [B, T-1, vocab_size]
                y is [B, T-1]
                loss_mask is [B, T-1]
        '''
        # input_ids and att_mask are [B, T]
        input_ids = batch['input_ids']
        att_mask  = batch['attn_mask']

        # if pos_ids is not provided, hf will add it automatically.
        pos_ids = batch.get('position_ids', None)
        if pos_ids is not None:
            pos_ids = pos_ids.to(att_mask.device)

        # feed data to model
        output = self.model_engine(input_ids=input_ids,
                                   attention_mask=att_mask,
                                   position_ids=pos_ids,
                                   use_cache=self.use_cache)

        # [B, T, vocab_size]
        every_token_logits = output.logits

        # label would be input_ids shifted by one (input_ids[:, 1:])
        # so the size is [B, T-1]
        target_ids = input_ids[:, 1:].contiguous()
        # remember it is an auto-regressive model: we use token [t] to predict token [t+1],
        # hence no need to predict last token's output (e.g., <eos>) and we remove it from logits.
        logits = every_token_logits[:, :-1, :].contiguous()

        # loss_mask is [B, T -1]
        loss_mask = batch['loss_mask'].contiguous()

        return logits, target_ids, loss_mask

    def eval_step(self, micro_batch):
        '''
           This function implements a single validation step per rank/gpu.
        '''
        # we need to split data into micro batches
        self.model_engine.eval()
        with torch.no_grad():
            # forward pass per gpu/rank
            logits, target_ids, loss_mask = self.forward(micro_batch)

            # compute loss pass
            loss = self.compute_loss(logits=logits, target_ids=target_ids, loss_mask=loss_mask)
            val_loss = loss.item()

        return {"loss": float(val_loss)}

    def train_step(self, micro_batch):
        '''
           This function implements a single training step per rank/gpu.
           The batch size for each gpu/rank should be micro_batch_size_per_gpu. 
           The DataLoader already yields micro-batches. 
        '''
        # make sure model is in training mode
        self.model_engine.train()

        # Don't need to zero_grad() here as ds handles gradient zeroing
        # internally after step() when gradient_accumulation_steps boundary is reached.

        # 1. forward pass per gpu/rank
        logits, target_ids, loss_mask = self.forward(micro_batch)

        # 3. compute loss pass
        loss = self.compute_loss(logits=logits, target_ids=target_ids, loss_mask=loss_mask)

        # 4. backward step
        # deepspeed aggregates gradients and only updates weights when accumulation_steps is reached.
        self.model_engine.backward(loss)

        # 5. optimizer step
        self.model_engine.step()

        # return loss
        train_loss = loss.item()
        return {"loss": float(train_loss)}