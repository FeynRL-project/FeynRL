from __future__ import annotations

import json
import os
from dataclasses import dataclass

import torch
from safetensors.torch import load_file
from peft import get_peft_model, LoraConfig


@dataclass(frozen=True)
class AdapterArtifact:
    adapter_dir: str
    peft_config_path: str
    weights_paths: list[str]


def wrap_with_lora(model, peft_config):
    if peft_config.peft_type != "lora":
        raise ValueError(f"Unsupported PEFT type: {peft_config.peft_type}")

    lora_config = LoraConfig(
        r=peft_config.lora_rank,
        lora_alpha=peft_config.lora_alpha,
        lora_dropout=peft_config.lora_dropout,
        target_modules=peft_config.lora_target_modules,
        task_type=peft_config.task_type,
    )
    return get_peft_model(model, lora_config)


def _discover_adapter_artifact(adapter_dir: str) -> AdapterArtifact:
    adapter_dir = os.path.abspath(adapter_dir)
    peft_config_path = os.path.join(adapter_dir, "peft_config.json")
    if not os.path.exists(peft_config_path):
        raise FileNotFoundError(f"Adapter artifact missing peft_config.json: {peft_config_path}")

    index_path = os.path.join(adapter_dir, "model.safetensors.index.json")
    single_path = os.path.join(adapter_dir, "model.safetensors")

    weights_paths: list[str] = []
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        shards = sorted(set(weight_map.values()))
        weights_paths = [os.path.join(adapter_dir, s) for s in shards]
        missing = [p for p in weights_paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"Adapter artifact shards missing: {missing[:3]}")
    elif os.path.exists(single_path):
        weights_paths = [single_path]
    else:
        raise FileNotFoundError(
            f"Adapter artifact missing weights: expected {single_path} or {index_path}"
        )

    return AdapterArtifact(
        adapter_dir=adapter_dir,
        peft_config_path=peft_config_path,
        weights_paths=weights_paths,
    )


def load_lora_adapter_weights_(model, adapter_dir: str, strict: bool = True):
    """
    Loads adapter-only tensors produced by this repo's checkpointing into a LoRA-wrapped model.
    The model must already be wrapped with PEFT (e.g., via wrap_with_lora()).
    """
    artifact = _discover_adapter_artifact(adapter_dir)

    # We don't currently reconstruct the LoRA config from peft_config.json because the
    # caller already constructed the wrapped model from the run config. Keep this file
    # for portability/debugging and for future compatibility checks.
    adapter_sd: dict[str, torch.Tensor] = {}
    for path in artifact.weights_paths:
        adapter_sd.update(load_file(path))

    missing, unexpected = model.load_state_dict(adapter_sd, strict=strict)
    return {"missing_keys": missing, "unexpected_keys": unexpected}

