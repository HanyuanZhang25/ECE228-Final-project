from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


LABEL_NAMES = ("Wake", "N1", "N2", "N3", "REM")


@dataclass(frozen=True)
class SplitArrays:
    X: np.ndarray
    y: np.ndarray
    subject_ids: np.ndarray
    recording_ids: np.ndarray


def read_processed_index(index_csv: Path) -> list[dict[str, str]]:
    with index_csv.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_split_arrays(index_csv: Path, split: str) -> SplitArrays:
    rows = [row for row in read_processed_index(index_csv) if row["split"] == split]
    if not rows:
        raise ValueError(f"No rows found for split={split!r} in {index_csv}")

    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    subject_parts: list[np.ndarray] = []
    recording_parts: list[np.ndarray] = []

    for row in rows:
        npz_path = Path(row["output"])
        with np.load(npz_path, allow_pickle=False) as data:
            X = data["X"].astype(np.float32, copy=False)
            y = data["y"].astype(np.int64, copy=False)
        X_parts.append(X)
        y_parts.append(y)
        subject_parts.append(np.full(y.shape[0], row["subject_id"]))
        recording_parts.append(np.full(y.shape[0], row["recording_id"]))

    return SplitArrays(
        X=np.concatenate(X_parts, axis=0),
        y=np.concatenate(y_parts, axis=0),
        subject_ids=np.concatenate(subject_parts, axis=0),
        recording_ids=np.concatenate(recording_parts, axis=0),
    )


class SleepEpochDataset(Dataset):
    def __init__(self, arrays: SplitArrays, context_size: int = 1):
        if context_size < 1 or context_size % 2 != 1:
            raise ValueError("context_size must be a positive odd integer")
        self.X = arrays.X
        self.y = arrays.y
        self.context_size = context_size
        self.window_indices = self._build_window_indices(arrays.recording_ids, context_size)

    @staticmethod
    def _build_window_indices(recording_ids: np.ndarray, context_size: int) -> np.ndarray | None:
        if context_size == 1:
            return None

        radius = context_size // 2
        windows = np.empty((recording_ids.shape[0], context_size), dtype=np.int64)
        unique_recordings = np.unique(recording_ids)

        for recording_id in unique_recordings:
            idx = np.flatnonzero(recording_ids == recording_id)
            for local_pos, center_idx in enumerate(idx):
                local_window = np.clip(
                    np.arange(local_pos - radius, local_pos + radius + 1),
                    0,
                    len(idx) - 1,
                )
                windows[center_idx] = idx[local_window]
        return windows

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # X is stored as (time, channels), i.e. (3000, 2).
        # With context_size > 1, x becomes (context, time, channels).
        if self.window_indices is None:
            x_np = self.X[index]
        else:
            x_np = self.X[self.window_indices[index]]
        x = torch.from_numpy(x_np)
        y = torch.tensor(self.y[index], dtype=torch.long)
        return x, y


def class_counts(y: np.ndarray, num_classes: int = 5) -> np.ndarray:
    return np.bincount(y, minlength=num_classes).astype(np.int64)


def inverse_frequency_class_weights(y: np.ndarray, num_classes: int = 5) -> torch.Tensor:
    counts = class_counts(y, num_classes=num_classes).astype(np.float32)
    total = counts.sum()
    weights = total / (num_classes * np.maximum(counts, 1.0))
    # Normalize around 1.0 so the loss scale stays tame.
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)
