import torch
from collections import deque
from typing import Dict, Any, List, Optional
from torch.utils.data import Dataset

# local imports
from misc.utils import ensure_1d, pad_1d_to_length

class ReplayBuffer(Dataset):
    '''
       Replay buffer for RL.
       It stores one trajectory per item (one sequence).
       max_size: optional hard cap on the number of stored items. Used
                 by run_rl_async.py to bound per-iter work in
                 prepare_training_batches and prevent the replay buffer from
                 growing without limit while training is slower than generation.
    '''
    def __init__(self,
                pad_token_id: int,
                max_seq_len: int,
                max_size: Optional[int] = None,
                ):

        if max_size is not None:
            self.items: Any = deque(maxlen=max_size)

        else:
            self.items = []

        self.max_size = max_size
        self.pad_token_id = int(pad_token_id)
        self.max_seq_len  = int(max_seq_len)
        # this shows the total number of action tokens which are not masked which
        # can be used for token-weighted scaling later.
        self.total_action_tokens = 0

    def add_batch_seqs(self, samples: List[Dict[str, Any]]) -> None:
        '''
            Add a batch of sequences to the replay buffer.
            Note that I have listed here everything that is collected in rollout_engine,
            but not all of them are used here. Each example in the batch has the following items in it:
            - iter: int                  --> [Not added to replay buffer for now]
            - policy_version: int
            - loaded_version: int        --> [Not added to replay buffer for now]
            - input_ids: torch.Tensor    --> [T]
            # token-aligned  -- NOT USED
                - token_rewards: torch.Tensor --> [T] [Not added to replay buffer for now]
                - token_zscores: torch.Tensor --> [T] [Not added to replay buffer for now]
                - token_masks: torch.Tensor   --> [T] [Not added to replay buffer for now]
                - token_dones: torch.Tensor   --> [T] [Not added to replay buffer for now]
                - token_old_logprobs: torch.Tensor --> [T] [Not added to replay buffer for now]
            # pred-aligned
                - pred_masks: torch.Tensor    --> [T] this is prediction aligned so no need to do any weired indexing
                - pred_dones: torch.Tensor    --> [T] this is prediction aligned so no need to do any weired indexing
                - pred_old_logprobs: torch.Tensor --> [T] this is prediction aligned so no need to do any weired indexing
                - pred_rewards: torch.Tensor --> [T] this is prediction aligned so no need to do any weired indexing
            - finish_reason: str         --> already used for done and mask
            - stop_reason: str           --> already used for done and mask
            - ended_on_eos: bool         --> already used for done and mask
            - response_ids: List[int]    --> input id already contains this
            - prompt_ids: List[int]      --> input id already contains this
            - response_text: str
            - response_len: int
        '''
        truncated_count = 0
        for sample in samples:
            if sample["response_len"] == 0:
                continue

            seq_len = sample["input_ids"].numel()
            # samples which are longer than max_seq_len are truncated,
            # otherwise there would be issuues with done, reward, mask, etc.
            if seq_len > self.max_seq_len:
                truncated_count += 1
                continue

            self.add(input_ids=sample["input_ids"],
                     rewards=sample["pred_rewards"],
                     zscores=sample["pred_zscores"],
                     masks=sample["pred_masks"],
                     dones=sample["pred_dones"],
                     old_logprobs=sample["pred_old_logprobs"],
                     policy_version=sample["policy_version"],
                     )

        if truncated_count > 0:
            print(f"[ReplayBuffer] {truncated_count}/{len(samples)} sequences truncated "
                  f"from prompt+response to max_seq_len={self.max_seq_len}. "
                  f"Consider reducing rollout max_tokens in rollouts or increasing max_seq_len in data configs.")

    def add(self,
            input_ids: torch.Tensor,
            rewards: torch.Tensor,
            zscores: torch.Tensor,
            masks: torch.Tensor,
            dones: torch.Tensor,
            old_logprobs: torch.Tensor,
            policy_version: int,
            )-> None:
        '''
            input_ids, rewards, zscores, mask, done, old_logprobs
            are all prediction aligned and [T].
        '''
        input_ids = ensure_1d(input_ids, "input_ids")
        rewards   = ensure_1d(rewards, "rewards")
        zscores   = ensure_1d(zscores, "zscores")
        masks     = ensure_1d(masks, "mask")
        dones     = ensure_1d(dones, "dones") # 1=eos, otherwise zero
        old_logps = ensure_1d(old_logprobs, "old_logprobs")

        # now create attn_masks
        attn_masks = torch.ones_like(input_ids)

        # all these should have the same length
        tensors = [input_ids, attn_masks, old_logps, masks, rewards, dones, zscores]

        all_len = {t.numel() for t in tensors}
        if len(all_len) != 1:
            raise ValueError(f"All tensors must have the same length; got lengths={sorted(all_len)}")

        # truncate to max_seq_len and save memory
        keep       = min(input_ids.numel(), self.max_seq_len)
        input_ids  = input_ids[:keep]
        attn_masks = attn_masks[:keep]
        old_logps  = old_logps[:keep]
        masks      = masks[:keep]
        rewards    = rewards[:keep]
        dones      = dones[:keep]
        zscores    = zscores[:keep]

        # everything stay in cpu
        self.items.append({"input_ids": input_ids.detach().cpu(),
                            "attn_masks": attn_masks.detach().cpu(),
                            "old_logps": old_logps.detach().cpu(),
                            "masks": masks.detach().cpu(),
                            "rewards": rewards.detach().cpu(),
                            "dones": dones.detach().cpu(),
                            "zscores": zscores.detach().cpu(),
                            "policy_version": int(policy_version),
                         })

        # Count only tokens we will ever train on
        self.total_action_tokens += int((masks > 0.5).sum().item())

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        '''
            Overwrite the default collate_fn to handle padding.
            Pads to target_len = min(max_len_in_batch, max_seq_len).
        '''
        if len(batch) == 0:
            raise ValueError("collate_fn received an empty batch")

        # calculate effective max_seq_len in the current batch
        # note data already truncated to max_seq_len in add()
        target_len = max(x["input_ids"].numel() for x in batch)

        # pad to batch_max_seq
        input_ids, attn_masks, old_logps = [], [], []
        masks, rewards, dones, zscores = [], [], [], []

        for x in batch:
            # pad everything to zero except for input_ids which should
            # be padded to pad_token_id
            # attn_mask vs mask:
            # attn_mask:  [1, 1, 1, 1, 1, 0, 0]   -> prompt + response which all real tokens would be 1 and pad would be zero
            # masks:      [0, 0, 1, 1, 0, 0, 0]   -> only valid prediction positions
            input_ids.append(pad_1d_to_length(x=x["input_ids"], pad_value=self.pad_token_id, target_len=target_len))
            attn_masks.append(pad_1d_to_length(x=x["attn_masks"], pad_value=0, target_len=target_len))
            old_logps.append(pad_1d_to_length(x=x["old_logps"], pad_value=0.0, target_len=target_len))
            masks.append(pad_1d_to_length(x=x["masks"], pad_value=0, target_len=target_len))
            rewards.append(pad_1d_to_length(x=x["rewards"], pad_value=0.0, target_len=target_len))
            dones.append(pad_1d_to_length(x=x["dones"], pad_value=0, target_len=target_len))
            zscores.append(pad_1d_to_length(x=x["zscores"], pad_value=0.0, target_len=target_len))

        # convert from list of [T] to [B, T]
        input_ids   = torch.stack(input_ids, dim=0)
        attn_masks  = torch.stack(attn_masks, dim=0)
        # old_logps may contain sentinel 1.0 at nan positions,
        # however mask is already zeroed there.
        old_logps   = torch.stack(old_logps, dim=0)
        masks       = torch.stack(masks, dim=0)
        rewards     = torch.stack(rewards, dim=0)
        dones       = torch.stack(dones, dim=0)
        zscores     = torch.stack(zscores, dim=0)

        # info for scaling later
        batch_action_tokens = int((masks > 0.5).sum().item())
        total_action_tokens = max(1, self.total_action_tokens)
        # this is per rank, this is not global. Should be revised outised this class.
        action_token_weight = float(batch_action_tokens) / float(total_action_tokens)

        return {
                "input_ids": input_ids, # [B, T]
                "attn_mask": attn_masks, # [B, T]
                "old_logprobs": old_logps, # [B, T]
                "mask": masks, # [B, T]
                "rewards": rewards, # [B, T]
                "done": dones, # [B, T]
                "zscore": zscores, # [B, T]
                "batch_action_tokens": batch_action_tokens, # scalar int
                "action_token_weight": action_token_weight, # scalar float
                }

    def __getitem__(self, idx) -> Dict[str, Any]:
        return self.items[idx]

    def __len__(self) -> int:
        return len(self.items)

    def reset(self) -> None:
        '''
            Clear the replay buffer for the next epoch.
        '''
        if self.max_size is not None:
            self.items = deque(maxlen=self.max_size)

        else:
            self.items = []
        self.total_action_tokens = 0

    def evict_stale(self, min_version: int) -> int:
        '''
            Remove samples generated by policies older than min_version.
            Returns the number of evicted ones.
        '''
        before = len(self.items)
        if isinstance(self.items, deque):
            self.items = deque((x for x in self.items if x["policy_version"] >= min_version), maxlen=self.items.maxlen,)

        else:
            self.items = [x for x in self.items if x["policy_version"] >= min_version]

        # Recount action tokens from remaining samples
        self.total_action_tokens = sum(int((x["masks"] > 0.5).sum().item()) for x in self.items)
        evicted = before - len(self.items)

        return evicted