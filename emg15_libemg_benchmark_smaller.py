"""Benchmark LibEMG statistical classifiers on the 15-channel AVE-Speech dataset.

This revision mirrors the official LibEMG feature pipeline: recordings are windowed,
standard feature groups are extracted via :class:`libemg.feature_extractor.FeatureExtractor`
and the learned normaliser is reused across validation and test splits.  The classifiers
therefore see the dense set of windows they were designed for instead of a single
summary vector per file, which substantially improves generalisation.
"""
from __future__ import annotations

import argparse
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import signal
from sklearn.metrics import accuracy_score

# ----------------------------------------------------------------------
# ✅ Path and CRC Fix
# ----------------------------------------------------------------------
LIBEMG_PATH = Path(__file__).resolve().parent / "libemg"
if str(LIBEMG_PATH) not in sys.path:
    sys.path.insert(0, str(LIBEMG_PATH))

try:
    import crc
    crc_module = types.SimpleNamespace()
    if not hasattr(crc, "Crc8"):
        class Crc8:
            def __init__(self, *args, **kwargs): pass
            def calculate(self, data): return 0
        crc.Crc8 = Crc8
    crc_module.CrcCalculator = getattr(crc, "CrcCalculator", None)
    crc_module.Crc8 = crc.Crc8
    sys.modules["crc.crc"] = crc_module
except Exception as exc:
    print(f"⚠️ CRC patch warning: {exc}")

# ----------------------------------------------------------------------
# ✅ Import EMGClassifier
# ----------------------------------------------------------------------
try:
    from libemg.emg_predictor import EMGClassifier
    from libemg.feature_extractor import FeatureExtractor
    from libemg.utils import get_windows
except ImportError as exc:
    raise SystemExit(
        f"❌ Could not import EMGClassifier: {exc}\n"
        "Make sure you run this script from inside 'libemg-main'."
    )

# ----------------------------------------------------------------------
# ✅ Import dataset config
# ----------------------------------------------------------------------
from emg15_dataset import EMGDataset15, MFSCConfig

# ----------------------------------------------------------------------
# 🔧 Configuration dataclass
# ----------------------------------------------------------------------
@dataclass(slots=True)
class FeatureExtractionConfig:
    sample_rate: int = 1000
    trim_start: int = 250
    window_size_ms: int = 200
    window_stride_ms: int = 50
    feature_group: str = "LS9"
    wamp_threshold: float = 0.02

    def window_size_samples(self) -> int:
        return max(1, int(round(self.sample_rate * self.window_size_ms / 1000)))

    def window_stride_samples(self) -> int:
        return max(1, int(round(self.sample_rate * self.window_stride_ms / 1000)))


# ----------------------------------------------------------------------
# ✅ Model hyperparameters
# ----------------------------------------------------------------------
MODEL_PARAMS: Dict[str, Dict[str, object]] = {
    "LDA": {"solver": "lsqr", "shrinkage": "auto"},
    "KNN": {"n_neighbors": 11, "weights": "distance", "metric": "minkowski", "p": 2, "n_jobs": -1},
    "SVM": {"kernel": "rbf", "C": 5.0, "gamma": "scale", "probability": True, "class_weight": "balanced"},
    "QDA": {"reg_param": 0.1},
    "RF": {
        "n_estimators": 400, "max_features": "sqrt", "min_samples_split": 2, "min_samples_leaf": 1,
        "bootstrap": True, "class_weight": "balanced_subsample", "n_jobs": -1, "random_state": 0,
    },
    "NB": {"var_smoothing": 1e-9},
    "GB": {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 3, "subsample": 0.9, "random_state": 0},
    "MLP": {
        "hidden_layer_sizes": (256, 128), "activation": "relu", "alpha": 1e-3,
        "learning_rate_init": 1e-3, "max_iter": 400, "early_stopping": True,
        "n_iter_no_change": 15, "random_state": 0,
    },
}

# ----------------------------------------------------------------------
# ✅ CLI arguments
# ----------------------------------------------------------------------
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=Path("runs/15vc/libemg_benchmark_results_advanced.csv"))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-train", type=int, default=4000)
    parser.add_argument("--max-val", type=int, default=600)
    parser.add_argument("--max-test", type=int, default=1200)
    parser.add_argument("--float32", action="store_true")
    return parser.parse_args(argv)


# ----------------------------------------------------------------------
# 🔧 Signal utilities
# ----------------------------------------------------------------------
def design_emg_filters(fs: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    filters = []
    for freq in (50, 150, 250, 350):
        filters.append(signal.iirnotch(freq, 30, fs))
    filters.append(signal.butter(4, [10 / (fs / 2), 400 / (fs / 2)], "bandpass"))
    return filters

def apply_filters(emg: np.ndarray, filters: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    filtered = emg
    for b, a in filters:
        padlen = max(len(a), len(b)) * 3
        if filtered.shape[1] <= padlen:
            continue
        filtered = signal.filtfilt(b, a, filtered, axis=1)
    return filtered


# ----------------------------------------------------------------------
# ✅ Dataset + processing
# ----------------------------------------------------------------------
def load_emg_from_file(path: Path) -> np.ndarray:
    mat = sio.loadmat(path)
    if "data" not in mat:
        raise KeyError(f"{path} missing 'data' variable")
    data = np.asarray(mat["data"], dtype=np.float32)
    if data.ndim == 3:
        return data[0].T
    if data.ndim == 2:
        return data.T
    return data.reshape(-1, 1).T


def prepare_split_features(
    dataset,
    limit,
    cfg: FeatureExtractionConfig,
    rng,
    shuffle,
    extractor: FeatureExtractor,
    feature_list: List[str],
    feature_params: Dict[str, float],
    normalizer=None,
    fit_normalizer: bool = False,
):
    indices = np.arange(len(dataset.file_list))
    if shuffle:
        rng.shuffle(indices)
    if limit > 0:
        indices = indices[:limit]

    filters = design_emg_filters(cfg.sample_rate)
    window_size = cfg.window_size_samples()
    window_stride = cfg.window_stride_samples()

    all_windows: List[np.ndarray] = []
    labels: List[int] = []
    sample_ids: List[int] = []

    if not fit_normalizer and normalizer is None:
        raise ValueError("A trained normalizer must be provided when fit_normalizer is False.")

    for i, dataset_index in enumerate(indices, 1):
        label_str, path = dataset.file_list[dataset_index]
        emg = load_emg_from_file(Path(path))
        if emg.shape[1] > cfg.trim_start:
            emg = emg[:, cfg.trim_start:]
        emg = emg - emg.mean(axis=1, keepdims=True)
        emg = apply_filters(emg, filters)
        if emg.shape[1] == 0:
            emg = np.zeros((emg.shape[0], window_size), dtype=emg.dtype)
        elif emg.shape[1] < window_size:
            pad = window_size - emg.shape[1]
            emg = np.pad(emg, ((0, 0), (0, pad)), mode="edge")

        windows = get_windows(emg.T, window_size, window_stride).astype(np.float32, copy=False)
        if windows.ndim == 2:
            windows = windows[np.newaxis, :, :]
        all_windows.append(windows)

        label_int = int(label_str)
        labels.extend([label_int] * len(windows))
        sample_id = i - 1
        sample_ids.extend([sample_id] * len(windows))
        if i % 250 == 0:
            print(f"   Processed {i}/{len(indices)} samples", flush=True)

    if not all_windows:
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            normalizer,
        )

    stacked_windows = np.concatenate(all_windows, axis=0)

    features, scaler = extractor.extract_features(
        feature_list,
        stacked_windows,
        feature_dic=feature_params,
        array=True,
        normalize=True,
        normalizer=None if fit_normalizer else normalizer,
        fix_feature_errors=True,
    )

    return (
        features,
        np.asarray(labels, dtype=np.int64),
        np.asarray(sample_ids, dtype=np.int64),
        scaler if fit_normalizer else normalizer,
    )


# ----------------------------------------------------------------------
# ✅ Benchmark + evaluation
# ----------------------------------------------------------------------
def majority_vote_accuracy(preds: np.ndarray, labels: np.ndarray, sample_ids: np.ndarray) -> float:
    if preds.size == 0 or sample_ids.size == 0:
        return float("nan")

    unique_ids = np.unique(sample_ids)
    majority_labels: List[int] = []
    majority_preds: List[int] = []

    for sample_id in unique_ids:
        mask = sample_ids == sample_id
        if not np.any(mask):
            continue
        window_votes = preds[mask]
        values, counts = np.unique(window_votes, return_counts=True)
        majority_preds.append(int(values[np.argmax(counts)]))
        majority_labels.append(int(labels[mask][0]))

    if not majority_labels:
        return float("nan")

    return accuracy_score(majority_labels, majority_preds)


def benchmark_classifier(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    test_ids: np.ndarray,
):
    params = MODEL_PARAMS.get(name, {}).copy()
    if "random_state" in params:
        params["random_state"] = random_state

    clf = EMGClassifier(name, model_parameters=params, random_seed=random_state)
    clf.fit({"training_features": X_train, "training_labels": y_train})

    train_pred, _ = clf.run(X_train)
    val_pred, _ = clf.run(X_val)
    test_pred, _ = clf.run(X_test)

    return {
        "train_window": accuracy_score(y_train, train_pred),
        "val_window": accuracy_score(y_val, val_pred),
        "test_window": accuracy_score(y_test, test_pred),
        "train_sample": majority_vote_accuracy(train_pred, y_train, train_ids),
        "val_sample": majority_vote_accuracy(val_pred, y_val, val_ids),
        "test_sample": majority_vote_accuracy(test_pred, y_test, test_ids),
    }


# ----------------------------------------------------------------------
# ✅ Main
# ----------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rng = np.random.default_rng(args.random_state)

    print("📦 Loading EMG dataset...", flush=True)
    cfg = MFSCConfig()
    fcfg = FeatureExtractionConfig(sample_rate=cfg.sample_rate, trim_start=cfg.trim_start)
    extractor = FeatureExtractor()
    feature_groups = extractor.get_feature_groups()
    if fcfg.feature_group not in feature_groups:
        raise ValueError(f"Unknown feature group '{fcfg.feature_group}'. Available: {sorted(feature_groups)}")
    feature_list = feature_groups[fcfg.feature_group]
    feature_params = {"WAMP_threshold": float(fcfg.wamp_threshold)}

    splits = {
        "train": EMGDataset15(str(args.dataset_root), split="train", cfg=cfg),
        "val": EMGDataset15(str(args.dataset_root), split="val", cfg=cfg),
        "test": EMGDataset15(str(args.dataset_root), split="test", cfg=cfg),
    }

    limits = {"train": args.max_train, "val": args.max_val, "test": args.max_test}
    arrays = {}
    normalizer = None

    for split, dataset in splits.items():
        fit_normalizer = split == "train"
        print(
            f"\n⚙️ Extracting LibEMG {fcfg.feature_group} features for {split} split "
            f"(limit={limits[split]})..."
        )
        X, y, sample_ids, returned_normalizer = prepare_split_features(
            dataset,
            limits[split],
            fcfg,
            rng,
            shuffle=(split == "train"),
            extractor=extractor,
            feature_list=feature_list,
            feature_params=feature_params,
            normalizer=normalizer,
            fit_normalizer=fit_normalizer,
        )
        if fit_normalizer:
            normalizer = returned_normalizer
        X = X.astype(np.float32, copy=False) if args.float32 else X
        arrays[split] = (X, y, sample_ids)
        print(f"✅ {split.capitalize()} ready → windows={X.shape[0]} | features={X.shape[1] if X.size else 0}")

    if normalizer is None:
        raise RuntimeError("Failed to compute feature normaliser from the training data.")

    X_train, y_train, train_ids = arrays["train"]
    X_val, y_val, val_ids = arrays["val"]
    X_test, y_test, test_ids = arrays["test"]

    models = ["LDA", "KNN", "SVM", "QDA", "RF", "NB", "GB", "MLP"]
    results: List[Dict[str, float]] = []

    def fmt_score(value: float) -> str:
        return "n/a" if np.isnan(value) else f"{value:.2%}"

    def to_percent(value: float) -> float:
        return float("nan") if np.isnan(value) else value * 100.0

    for m in models:
        print(f"\n▶ Running {m} classifier...")
        start = time.perf_counter()
        metrics = benchmark_classifier(
            m,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            args.random_state,
            train_ids,
            val_ids,
            test_ids,
        )
        elapsed = time.perf_counter() - start
        print(
            f"{m}: Train_win={fmt_score(metrics['train_window'])} | Train_file={fmt_score(metrics['train_sample'])} | "
            f"Val_win={fmt_score(metrics['val_window'])} | Val_file={fmt_score(metrics['val_sample'])} | "
            f"Test_win={fmt_score(metrics['test_window'])} | Test_file={fmt_score(metrics['test_sample'])} | "
            f"Time={elapsed:.1f}s"
        )
        results.append(
            {
                "Model": m,
                "Train (window %)": to_percent(metrics["train_window"]),
                "Train (file %)": to_percent(metrics["train_sample"]),
                "Val (window %)": to_percent(metrics["val_window"]),
                "Val (file %)": to_percent(metrics["val_sample"]),
                "Test (window %)": to_percent(metrics["test_window"]),
                "Test (file %)": to_percent(metrics["test_sample"]),
                "Time (s)": elapsed,
            }
        )

    df = pd.DataFrame(results).sort_values("Test (file %)", ascending=False)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print("\n🏁 Summary:")
    print(df)
    print(f"\n💾 Saved to: {args.output_csv.resolve()}")
    return 0


# ----------------------------------------------------------------------
# ✅ Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        raise
