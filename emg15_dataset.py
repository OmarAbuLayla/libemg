"""Dataset utilities for training EMG models on the 15 virtual-channel corpus.

The loader mirrors the directory layout used by the legacy six-channel
pipeline (Train/Val/Test subject folders) but removes the aggressive global
normalisation that previously washed away channel-dependent variance.  Each
sample is filtered, transformed into a log-mel representation and then
standardised on a per-channel basis so that polarity and relative amplitude
information are preserved for the network.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import librosa
import numpy as np
import scipy.io as sio
from scipy import signal
import torch
from torch.utils.data import Dataset


__all__ = ["EMGDataset15", "build_dataloaders", "MFSCConfig"]


@dataclass
class MFSCConfig:
    """Configuration parameters for the EMG mel-filterbank front-end."""

    sample_rate: int = 1000
    n_mels: int = 36
    n_fft: int = 256
    hop_length: int = 50
    max_frames: int = 36
    trim_start: int = 250
    per_channel_normalise: bool = True


def _design_filters(fs: int) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Pre-compute the notch + bandpass filters used for EMG denoising."""

    filters = {
        "notch_50": signal.iirnotch(50, 30, fs),
        "notch_150": signal.iirnotch(150, 30, fs),
        "notch_250": signal.iirnotch(250, 30, fs),
        "notch_350": signal.iirnotch(350, 30, fs),
        "bandpass": signal.butter(4, [10 / (fs / 2), 400 / (fs / 2)], "bandpass"),
    }
    return filters


def _apply_filters(emg: np.ndarray, filters: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Apply the cascaded notch/band-pass filters along the temporal axis."""

    padlen = max(max(len(b), len(a)) for b, a in filters.values()) * 3
    if emg.shape[1] < padlen:
        # Signal too short for zero-phase filtering; return a copy to avoid
        # mutating the original array downstream.
        return emg.copy()

    filtered = emg
    for b, a in filters.values():
        filtered = signal.filtfilt(b, a, filtered, axis=1)
    return filtered


def _ensure_sample_axis(emg: np.ndarray) -> np.ndarray:
    """Normalise loaded matrices to have a leading singleton sample axis."""

    if emg.ndim == 3:
        return emg.astype(np.float32, copy=False)
    if emg.ndim == 2:
        return emg[np.newaxis, :, :].astype(np.float32, copy=False)
    if emg.ndim == 1:
        return emg.reshape(1, -1, 1).astype(np.float32, copy=False)
    raise ValueError(f"Unexpected EMG tensor shape {emg.shape}")


def _load_emg(path: str) -> np.ndarray:
    """Load an EMG recording from a MATLAB .mat file."""

    mat = sio.loadmat(path)
    if "data" not in mat:
        raise KeyError(f"File {path} does not contain a 'data' variable")
    return _ensure_sample_axis(np.asarray(mat["data"], dtype=np.float32))


def _compute_mfsc(
    emg: np.ndarray,
    cfg: MFSCConfig,
    filters: Dict[str, Tuple[np.ndarray, np.ndarray]] | None = None,
) -> np.ndarray:
    """Compute log-mel spectrogram features for every channel."""

    if emg.shape[1] > cfg.trim_start:
        emg = emg[:, cfg.trim_start :, :]

    # Remove channel-wise DC components before filtering so that polarity is
    # preserved but slow drift does not dominate the spectrum.
    emg = emg - emg.mean(axis=1, keepdims=True)

    filters = filters or _design_filters(cfg.sample_rate)
    emg = _apply_filters(emg, filters)

    n_samples, n_timesteps, n_channels = emg.shape
    specs: List[np.ndarray] = []

    for ch in range(n_channels):
        channel_spec = []
        for sample in range(n_samples):
            signal_i = emg[sample, :, ch]
            if np.allclose(signal_i, 0):
                channel_spec.append(np.zeros((cfg.n_mels, cfg.max_frames), dtype=np.float32))
                continue

            n_fft = min(cfg.n_fft, len(signal_i))
            hop = min(cfg.hop_length, max(1, len(signal_i) // max(cfg.max_frames - 1, 1)))
            mel = librosa.feature.melspectrogram(
                y=np.asfortranarray(signal_i),
                sr=cfg.sample_rate,
                n_fft=n_fft,
                hop_length=hop,
                n_mels=cfg.n_mels,
                power=2.0,
            )
            mel = librosa.power_to_db(mel + 1e-8)
            if mel.shape[1] < cfg.max_frames:
                pad = cfg.max_frames - mel.shape[1]
                mel = np.pad(mel, ((0, 0), (0, pad)), mode="edge")
            elif mel.shape[1] > cfg.max_frames:
                mel = mel[:, : cfg.max_frames]
            channel_spec.append(mel.astype(np.float32))
        specs.append(np.stack(channel_spec, axis=0))

    stacked = np.stack(specs, axis=1)  # (samples, channels, n_mels, frames)

    if cfg.per_channel_normalise:
        mean = stacked.mean(axis=(0, 2, 3), keepdims=True)
        std = stacked.std(axis=(0, 2, 3), keepdims=True).clip(min=1e-6)
        stacked = (stacked - mean) / std

    return stacked


class EMGDataset15(Dataset):
    """PyTorch dataset for the 15-channel EMG corpus."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        cfg: MFSCConfig | None = None,
    ) -> None:
        super().__init__()
        self.root = os.path.abspath(root)
        self.split = split
        self.cfg = cfg or MFSCConfig()
        self._filters = _design_filters(self.cfg.sample_rate)
        self.file_list = self._discover_files()
        self._cache: Dict[int, Tuple[torch.Tensor, int]] = {}

        if not self.file_list:
            raise RuntimeError(
                f"Found no .mat files for split '{split}' under {self.root}. "
                "Expected layout: <root>/<Split>/EMG/<subject>/<session>/<label>.mat"
            )

    # ------------------------------------------------------------------
    def _discover_files(self) -> List[Tuple[str, str]]:
        split_dir = os.path.join(self.root, self.split.capitalize(), "EMG")
        if not os.path.isdir(split_dir):
            return []

        samples: List[Tuple[str, str]] = []
        for subject in sorted(os.listdir(split_dir)):
            subj_dir = os.path.join(split_dir, subject)
            if not os.path.isdir(subj_dir):
                continue
            for session in sorted(os.listdir(subj_dir)):
                sess_dir = os.path.join(subj_dir, session)
                if not os.path.isdir(sess_dir):
                    continue
                for fname in sorted(os.listdir(sess_dir)):
                    if not fname.endswith(".mat"):
                        continue
                    label = os.path.splitext(fname)[0]
                    samples.append((label, os.path.join(sess_dir, fname)))
        return samples

    # ------------------------------------------------------------------
    def __len__(self) -> int:  # type: ignore[override]
        return len(self.file_list)

    # ------------------------------------------------------------------
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:  # type: ignore[override]
        if index in self._cache:
            return self._cache[index]

        label_str, path = self.file_list[index]
        emg = _load_emg(path)
        features = _compute_mfsc(emg, self.cfg, self._filters)[0]  # remove sample axis

        # Convert the textual label to an integer class id.
        label = int(label_str)
        sample = (torch.from_numpy(features), label)
        self._cache[index] = sample
        return sample


def build_dataloaders(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 0,
    cfg: MFSCConfig | None = None,
) -> Dict[str, torch.utils.data.DataLoader]:
    """Create train/val/test dataloaders for the 15-channel dataset."""

    splits = ["train", "val", "test"]
    datasets = {}
    for split in splits:
        dataset = EMGDataset15(data_root, split=split, cfg=cfg)
        print(f"Loaded {split} split with {len(dataset)} samples from {data_root}")
        datasets[split] = dataset
    loaders = {
        split: torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )
        for split, dataset in datasets.items()
    }
    return loaders