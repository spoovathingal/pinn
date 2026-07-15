"""Dataset loading utilities for PINN training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

TensorDict = dict[str, torch.Tensor]


def load_npz_dataset(path: str | Path, device: torch.device | str = "cpu") -> TensorDict:
    """Load an .npz dataset and convert all arrays to float32 tensors."""
    data = np.load(Path(path))
    return {
        key: torch.as_tensor(data[key], dtype=torch.float32, device=device)
        for key in data.files
    }
