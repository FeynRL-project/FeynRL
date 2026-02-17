import torch
from typing import Dict, Any, List, Optional
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

    def add_batch_seqs(self, samples: List[Dict[str, Any]]) -> None:
        '''
            Add a batch of sequences to the replay buffer.
            Note that I have listed here everything that is collected in rollout_engine,
            but not all of them are used here.
            - iter: int                  --> [Not added to replay buffer for now]
            - policy_version: int        --> [Not added to replay buffer for now]
            - loaded_version: int        --> [Not added to replay buffer for now]
            - input_ids: torch.Tensor    --> [T]
            - rewards: torch.Tensor      --> [T]
            - zscores: torch.Tensor      --> [T] same as rewards if only one sample per prompt
            - token_mask: torch.Tensor   --> [T] [Not added to replay buffer for now]
            - token_done: torch.Tensor   --> [T] [Not added to replay buffer for now]
            - token_old_logprobs: torch.Tensor --> [T] [Not added to replay buffer for now]
            - pred_mask: torch.Tensor    --> [T] this is prediction aligned so no need to do any weired indexing
            - pred_done: torch.Tensor    --> [T] this is prediction aligned so no need to do any weired indexing
            - pred_old_logprobs: torch.Tensor --> [T] this is prediction aligned so no need to do any weired indexing
            - finish_reason: str         --> already used for done and mask
            - stop_reason: str           --> already used for done and mask
            - ended_on_eos: bool         --> already used for done and mask
            - response_ids: List[int]    --> input id already contains this
            - prompt_ids: List[int]      --> input id already contains this
            - response_text: str
            - response_len: int
        '''
        for sample in samples:
            if sample["response_len"] == 0:
                continue

            self.add(input_ids=sample["input_ids"],
                     rewards=sample["rewards"],
                     zscores=sample["pred_zscores"],
                     masks=sample["pred_masks"],
                     dones=sample["pred_dones"],
                     old_logprobs=sample["pred_old_logprobs"],
                     )

    def add(self,
            input_ids: torch.Tensor,
            rewards: torch.Tensor,
            zscores: torch.Tensor,
            masks: torch.Tensor,
            dones: torch.Tensor,
            old_logprobs: torch.Tensor,
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
        keep = min(input_ids.numel(), self.max_seq_len)
        input_ids   = input_ids[:keep]
        attn_masks   = attn_masks[:keep]
        old_logps   = old_logps[:keep]
        masks = masks[:keep]
        rewards     = rewards[:keep]
        dones       = dones[:keep]
        zscores     = zscores[:keep]

        # Keep on CPU; dataLoader can pin_memory for faster H2D.
        self.items.append({
            "input_ids": input_ids.detach().cpu(),
            "attn_masks": attn_masks.detach().cpu(),
            "old_logps": old_logps.detach().cpu(),
            "masks": masks.detach().cpu(),
            "rewards": rewards.detach().cpu(),
            "dones": dones.detach().cpu(),
            "zscores": zscores.detach().cpu(),
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
        self.items = []
        self.total_action_tokens = 0