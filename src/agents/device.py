"""PyTorch device selection helpers for Apple Silicon training."""

from __future__ import annotations


def get_torch_device(prefer_mps: bool = True) -> str:
    """Return the best available PyTorch device name for SAC training."""

    try:
        import torch
    except ImportError:
        return "cpu"

    mps_backend = getattr(torch.backends, "mps", None)
    if prefer_mps and mps_backend is not None and mps_backend.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
