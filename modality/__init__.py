from modality.base import ModalityAdapter
from modality.text import TextOnlyAdapter

_ADAPTER_REGISTRY: dict[str, type[ModalityAdapter]] = {
    "text": TextOnlyAdapter,
}


def build_adapter(name: str) -> ModalityAdapter:
    """Construct a ModalityAdapter by registry name.

    Raises ValueError for unknown names so callers get a clear message
    rather than a silent fallback to the wrong adapter.
    """
    cls = _ADAPTER_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown modality adapter '{name}'. "
            f"Supported: {list(_ADAPTER_REGISTRY)!r}."
        )
    return cls(tokenizer=None)


__all__ = ["ModalityAdapter", "TextOnlyAdapter", "build_adapter"]
