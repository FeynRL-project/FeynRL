import torch
from typing import Dict, Optional, Any, List
from torch.utils.data import Dataset

# local imports
from misc.utils import ensure_1d, pad_1d_to_length

class ReplayBuffer(Dataset):
    '''
       Replay buffer for RL.
       It stores one trajectory per item (one sequence).
    '''
    def __init__(self,
                pad_token_id: int,
                max_seq_len: int
                ):

        self.items: List[Dict[str, Optional[torch.Tensor]]] = []
        self.pad_token_id = int(pad_token_id)
        self.max_seq_len  = int(max_seq_len)
        # this shows the total number of action tokens which are not masked which
        # can be used for token-weighted scaling later.
        self.total_action_tokens = 0

    def add(self,
            generated_token_ids: torch.Tensor,
            generation_logps: torch.Tensor,
            prompt_token_ids: torch.Tensor,
            rewards: torch.Tensor,
            dones: torch.Tensor,
            v_old: Optional[torch.Tensor] = None,
            )-> None:
        '''
            dones: 1=eos, 0=not eos
        '''
        input_ids =  ensure_1d(input_ids, "input_ids")
        attn_mask = ensure_1d(attn_mask, "attn_mask")
        old_logps = ensure_1d(old_logps, "old_logps")
        token_masks = ensure_1d(token_masks, "token_masks")
        rewards = ensure_1d(rewards, "rewards")
        dones = ensure_1d(dones, "dones")

        if v_old is not None:
            v_old = self.ensure_1d(v_old, "v_old")

        # all these should have the same length
        tensors = [input_ids, attn_mask, old_logps, token_masks, rewards, dones]
        if v_old is not None:
            tensors.append(v_old)

        all_len = {t.numel() for t in tensors}
        if len(all_len) != 1:
            raise ValueError(f"All tensors must have the same length; got lengths={sorted(all_len)}")

        # truncate to max_seq_len and save memory
        keep = min(input_ids.numel(), self.max_seq_len)
        input_ids   = input_ids[:keep]
        attn_mask   = attn_mask[:keep]
        old_logps   = old_logps[:keep]
        token_masks = token_masks[:keep]
        rewards     = rewards[:keep]
        dones       = dones[:keep]
        if v_old is not None:
            v_old = v_old[:keep]

        # Keep on CPU; dataLoader can pin_memory for faster H2D.
        self.items.append({
            "input_ids": input_ids.detach().cpu(),
            "attn_mask": attn_mask.detach().cpu(),
            "old_logps": old_logps.detach().cpu(),
            "token_masks": token_masks.detach().cpu(),
            "rewards": rewards.detach().cpu(),
            "dones": dones.detach().cpu(),
            "v_old": v_old.detach().cpu() if v_old is not None else None,
                        })

        # Count only tokens we will ever train on
        self.total_action_tokens += int((token_masks > 0.5).sum().item())

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
        input_ids, attn_mask, old_logps = [], [], []
        token_masks, rewards, dones = [], [], []
        v_old_list = []
        empty_v_count = 0

        for x in batch:
            # pad everything to zero except for input_ids which should
            # be padded to pad_token_id
            input_ids.append(self.pad_to_len(x=x["input_ids"], pad_value=self.pad_token_id, target_len=target_len))
            attn_mask.append(self.pad_to_len(x=x["attn_mask"], pad_value=0, target_len=target_len))
            old_logps.append(self.pad_to_len(x=x["old_logps"], pad_value=0.0, target_len=target_len))
            token_masks.append(self.pad_to_len(x=x["token_masks"], pad_value=0, target_len=target_len))
            rewards.append(self.pad_to_len(x=x["rewards"], pad_value=0.0, target_len=target_len))
            dones.append(self.pad_to_len(x=x["dones"], pad_value=0, target_len=target_len))

            # if it is None, v_old_list will append None too
            if x["v_old"] is not None:
                v_old_list.append(self.pad_to_len(x["v_old"], pad_value=0.0, target_len=target_len))

            else:
                empty_v_count += 1

        # convert from list of [T] to [B, T]
        input_ids   = torch.stack(input_ids, dim=0)
        attn_mask   = torch.stack(attn_mask, dim=0)
        old_logps   = torch.stack(old_logps, dim=0)
        token_masks = torch.stack(token_masks, dim=0)
        rewards     = torch.stack(rewards, dim=0)
        dones       = torch.stack(dones, dim=0)

        if empty_v_count == len(batch):
            v_old = None

        elif empty_v_count== 0:
            v_old = torch.stack(v_old_list, dim=0)

        else:
            raise ValueError("Mixed None/non-None v_old inside the same batch")

        # info for scaling later
        batch_action_tokens = int((token_masks > 0.5).sum().item())
        total_action_tokens = max(1, self.total_action_tokens)
        # this is per rank, this is not global. Should be revised outised this class.
        action_token_weight = float(batch_action_tokens) / float(total_action_tokens)

        return {
                "input_ids": input_ids, # [B, T]
                "attn_mask": attn_mask, # [B, T]
                "old_logps": old_logps, # [B, T]
                "token_masks": token_masks, # [B, T]
                "rewards": rewards, # [B, T]
                "dones": dones, # [B, T]
                "v_old": v_old, # [B, T] or None
                "batch_action_tokens": batch_action_tokens, # scalar int
                "action_token_weight": action_token_weight, # scalar float
                }

    def __getitem__(self, idx) -> Dict[str, Any]:
        return self.items[idx]

    def __len__(self) -> int:
        return len(self.items)

