from __future__ import annotations

from typing import Callable

_REGISTRY: dict[str, Callable] = {}


def register(name: str) -> Callable:
    def decorator(fn: Callable) -> Callable:
        _REGISTRY[name] = fn
        return fn
    return decorator


def get_loader(name: str) -> Callable:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model_class '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_loaders() -> list[str]:
    return list(_REGISTRY)
