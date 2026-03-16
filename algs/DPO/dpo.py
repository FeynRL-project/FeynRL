import torch
import torch.nn.functional as F

class DPO:
    def __init__(self, model_engine,
                 ref_model_engine,
                 optimizer,
                 beta,
                 normalize_loss=False):

        self.model_engine = model_engine
        self.ref_model_engine = ref_model_engine
        # ref model would not be updated, so we put it in eval mode from get-go
        self.ref_model_engine.eval()
        self.optimizer = optimizer
        # normalize_loss is accepted for interface consistency with SFT but is unused:
        # DPO loss is inherently per-example, length-normalized log-ratios and mean
        # over the batch is the correct reduction regardless of this flag.
        self.beta = beta

        # use cross entropy loss
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="none")

    def compute_loss(self, logits, ref_logprobs, target_ids, loss_mask):
        '''
            This implements length-normalized dpo loss.
            logits: [2B, T-1, vocab_size]
            ref_logprobs: [2B, T-1]  (already reduced from ref_logits in forward)
            target_ids: [2B, T-1]
            loss_mask: [2B, T-1]
        '''
        two_B, T_minus_1, vocab_size = logits.shape

        # Compute per-token log-probs in float32 to avoid bf16/fp16 quantization
        # cross_entropy returns -logprobs, so we take negative
        # logits: [2B, T-1, vocab_size] -> [2B * (T-1), vocab_size]
        # target_ids: [2B, T-1] -> [2B * (T-1)]
        target_ids = target_ids.view(-1)
        neg_logprobs = self.cross_entropy(logits.to(torch.float32).view(-1, vocab_size), target_ids)
        logprobs = -neg_logprobs.view(two_B, T_minus_1)

        # Rows are interleaved: [chosen0, rejected0, chosen1, rejected1, ...] as
        # torch.stack([chosen, rejected], dim=0) is used to stack them, so
        # even rows(0::2) = chosen, odd rows(1::2) = rejected
        # [2B, T-1] -> [B, T-1]
        chosen_logprobs   = logprobs[0::2]
        rejected_logprobs = logprobs[1::2]
        ref_chosen_logprobs   = ref_logprobs[0::2]
        ref_rejected_logprobs = ref_logprobs[1::2]
        # [2B, T-1] -> [B, T-1]
        chosen_mask   = loss_mask[0::2].to(torch.float32)
        rejected_mask = loss_mask[1::2].to(torch.float32)

        # Per-token logratios where masked padding/prompt positions are zeroed out.
        # all are [B, T-1]
        chosen_token_logratios   = chosen_mask * (chosen_logprobs - ref_chosen_logprobs)
        rejected_token_logratios = rejected_mask * (rejected_logprobs - ref_rejected_logprobs)

        # sum over the sequence length dimension to get length-normalized logratios per example
        # [B, T-1] -> [B]
        len_chosen   = chosen_mask.sum(dim=1).clamp(min=1.0)
        len_rejected = rejected_mask.sum(dim=1).clamp(min=1.0)

        # chosen_token_logratios.sum(dim=1): [B, T-1] -> [B]
        chosen_rewards   = chosen_token_logratios.sum(dim=1) / len_chosen
        rejected_rewards = rejected_token_logratios.sum(dim=1) / len_rejected

        # dpo loss: -log sigmoid(beta * (chosen_reward - rejected_reward))
        loss = -F.logsigmoid(self.beta * (chosen_rewards - rejected_rewards)).mean()

        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError(f"DPO loss is NaN or Inf: {loss.item()}")

        metrics = {"loss": float(loss.item()),
                   "chosen_rewards": float(chosen_rewards.mean().item()),
                   "rejected_rewards": float(rejected_rewards.mean().item()),
                   "reward_accuracies": float((chosen_rewards > rejected_rewards).float().mean().item()),
                  }
        return loss, metrics

    def forward(self, batch):
        '''
            batch[input_ids/attn_mask]: [B, 2, T]
            batch[loss_mask]: [B, 2, T-1]
        '''
        # since torch.stack([chosen, rejected], dim=0) is used to stack them, data
        # are interleaved as [chosen0, rejected0, chosen1, rejected1, ...]
        if batch['input_ids'].dim() != 3 or batch['input_ids'].shape[1] != 2:
            raise ValueError(f"DPO expects input_ids of shape [B, 2, T], got {list(batch['input_ids'].shape)}")

        B, _, T = batch['input_ids'].shape
        # [B, 2, T] -> [2B, T]
        input_ids = batch['input_ids'].view(-1, T)
        att_mask  = batch['attn_mask'].view(-1, T)
        # [B, 2, T-1] -> [2B, T-1]
        loss_mask = batch['loss_mask'].view(-1, batch['loss_mask'].shape[-1])

        # if pos_ids is not provided, hf will add it automatically.
        pos_ids = batch.get('position_ids', None)
        if pos_ids is not None:
            # [B, 2, T] -> [2B, T]
            pos_ids = pos_ids.view(-1, T).to(att_mask.device)

        # feed data to model
        output = self.model_engine(input_ids=input_ids,
                                   attention_mask=att_mask,
                                   position_ids=pos_ids,
                                   use_cache=False)

        # [2B, T, vocab_size] -> [2B, T-1, vocab_size]
        # remember we use token t to predict token t+1, hence no need to predict last
        # token's output (e.g., <eos>) and we remove it from logits.
        logits = output.logits[:, :-1, :].contiguous()

        # label would be input_ids shifted by one
        # [2B, T] -> [2B, T-1]
        target_ids = input_ids[:, 1:].contiguous()

        # Compute ref logprobs inside no_grad and reduce to [2B, T-1] immediately.
        # This frees the full [2B, T-1, vocab_size] ref_logits tensor.
        with torch.no_grad():
            ref_output = self.ref_model_engine(input_ids=input_ids,
                                               attention_mask=att_mask,
                                               position_ids=pos_ids,
                                               use_cache=False)

            ref_logits = ref_output.logits[:, :-1, :].contiguous()
            two_B, T_minus_1, v = ref_logits.shape
            # ref_logits: [2B, T-1, vocab_size] -> [2B * (T-1), vocab_size]
            # target_ids: [2B, T-1] -> [2B * (T-1)]
            neg_ref_logprobs = self.cross_entropy(ref_logits.to(torch.float32).view(-1, v),
                                                  target_ids.view(-1))
            ref_logprobs = -neg_ref_logprobs.view(two_B, T_minus_1)

        return logits, ref_logprobs, target_ids, loss_mask

    def eval_step(self, micro_batch):
        '''
           This implements a single validation step per rank/gpu.
           Setting model to eval mode and torch.no_grad() context are done in main.
        '''
        logits, ref_logprobs, target_ids, loss_mask = self.forward(micro_batch)
        loss, metrics = self.compute_loss(logits=logits,
                                          ref_logprobs=ref_logprobs,
                                          target_ids=target_ids,
                                          loss_mask=loss_mask)

        return metrics

    def train_step(self, micro_batch):
        '''
           This implements a single training step per rank/gpu
           for given micro_batch_size_per_gpu.
        '''
        # 1. forward pass per gpu/rank
        # chosen and rejected data are stacked as [B, 2, T]
        logits, ref_logprobs, target_ids, loss_mask = self.forward(micro_batch)

        # 2. compute loss
        loss, metrics = self.compute_loss(logits=logits,
                                          ref_logprobs=ref_logprobs,
                                          target_ids=target_ids,
                                          loss_mask=loss_mask)

        # 3. backward step
        # deepspeed aggregates gradients and only updates weights when accumulation_steps is reached.
        self.model_engine.backward(loss)

        # 4. optimizer step
        self.model_engine.step()

        return metrics