"""Benchmark LibEMG statistical classifiers on a curated subset of the 15-channel EMG dataset.

This script performs an end-to-end classical machine-learning benchmark that mirrors the
LibEMG EMGClassifier API while providing a significantly richer feature engineering and
pre-processing pipeline than the original minimal example.

Enhancements over the original version include:
    * Feature extraction directly from the cleaned raw EMG waveforms (RMS, MAV, WL, ZC, SSC, etc.).
    * Per-channel spectral descriptors (mean/median frequency, spectral entropy).
    * Channel-interaction cues through correlation features.
    * Class balancing via simple over-sampling to mitigate skewed training splits.
    * Robust standardisation followed by PCA retaining >99% variance (capped for efficiency).
    * Tuned hyper-parameters for each LibEMG statistical classifier.
    * Clear train/validation/test accuracy summaries and CSV logging.
"""
from __future__ import annotations

import argparse
import sys
import time
import types
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import signal, stats
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------
# ✅ Path and CRC Fix (Option 1)
# ----------------------------------------------------------------------
LIBEMG_PATH = Path(__file__).resolve().parent / "libemg"
if str(LIBEMG_PATH) not in sys.path:
    sys.path.insert(0, str(LIBEMG_PATH))

try:
    import crc

    crc_module = types.SimpleNamespace()
    if not hasattr(crc, "Crc8"):
        class Crc8:  # type: ignore
            def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - fallback
                pass

            def calculate(self, data):  # pragma: no cover - fallback
                return 0

        crc.Crc8 = Crc8
    crc_module.CrcCalculator = getattr(crc, "CrcCalculator", None)
    crc_module.Crc8 = crc.Crc8
    sys.modules["crc.crc"] = crc_module
except Exception as exc:  # pragma: no cover - defensive logging
    print(f"⚠️ CRC patch warning: {exc}")

# ----------------------------------------------------------------------
# ✅ Import EMGClassifier directly from libemg.emg_predictor
# ----------------------------------------------------------------------
try:
    from libemg.emg_predictor import EMGClassifier
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        f"❌ Could not import EMGClassifier from libemg.emg_predictor: {exc}\n"
        "Make sure you run this script from inside the 'libemg' repository root."
    )

# ----------------------------------------------------------------------
# ✅ Import dataset config (provides sampling rate information)
# ----------------------------------------------------------------------
from emg15_dataset import EMGDataset15, MFSCConfig


# ----------------------------------------------------------------------
# 🔧 Configuration dataclasses and defaults
# ----------------------------------------------------------------------
@dataclass(slots=True)
class FeatureExtractionConfig:
    """Configuration controlling the handcrafted feature pipeline."""

    sample_rate: int = 1000
    trim_start: int = 250
    zc_threshold_ratio: float = 0.05
    wamp_threshold_ratio: float = 0.05
    smoothing_window_ms: int = 25
    pca_variance: float = 0.99
    pca_max_components: int = 128
    correlation_weight: float = 0.5  # scales correlation features relative to per-channel stats


# Enhanced: tuned model parameters for LibEMG classifiers.
MODEL_PARAMS: Dict[str, Dict[str, object]] = {
    "LDA": {"solver": "lsqr", "shrinkage": "auto"},
    "KNN": {"n_neighbors": 11, "weights": "distance", "metric": "minkowski", "p": 2, "n_jobs": -1},
    "SVM": {
        "kernel": "rbf",
        "C": 5.0,
        "gamma": "scale",
        "probability": True,
        "class_weight": "balanced",
    },
    "QDA": {"reg_param": 0.1},
    "RF": {
        "n_estimators": 400,
        "max_features": "sqrt",
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "bootstrap": True,
        "class_weight": "balanced_subsample",
        "n_jobs": -1,
        "random_state": 0,
    },
    "NB": {"var_smoothing": 1e-9},
    "GB": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 3,
        "subsample": 0.9,
        "random_state": 0,
    },
    "MLP": {
        "hidden_layer_sizes": (256, 128),
        "activation": "relu",
        "alpha": 1e-3,
        "learning_rate_init": 1e-3,
        "max_iter": 400,
        "early_stopping": True,
        "n_iter_no_change": 15,
        "random_state": 0,
    },
}


# ----------------------------------------------------------------------
# ✅ CLI argument parsing
# ----------------------------------------------------------------------
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=Path("runs/15vc/libemg_benchmark_results_advanced.csv"))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-train", type=int, default=4000, help="Maximum number of training samples to use")
    parser.add_argument("--max-val", type=int, default=600, help="Maximum number of validation samples to use")
    parser.add_argument("--max-test", type=int, default=1200, help="Maximum number of test samples to use")
    parser.add_argument("--float32", action="store_true", help="Return features as float32 instead of float64")
    parser.add_argument("--no-pca", action="store_true", help="Disable PCA dimensionality reduction")
    return parser.parse_args(argv)


# ----------------------------------------------------------------------
# 🔧 EMG signal utilities (filtering + feature extraction)
# ----------------------------------------------------------------------
def design_emg_filters(fs: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Design notch and band-pass filters used to denoise the EMG channels."""

    filters: List[Tuple[np.ndarray, np.ndarray]] = []
    for freq in (50, 150, 250, 350):
        filters.append(signal.iirnotch(freq, 30, fs))
    filters.append(signal.butter(4, [10 / (fs / 2), 400 / (fs / 2)], "bandpass"))
    return filters


def apply_filters(emg: np.ndarray, filters: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Apply cascaded zero-phase filtering along the temporal axis."""

    filtered = emg
    for b, a in filters:
        padlen = max(len(a), len(b)) * 3
        if filtered.shape[1] <= padlen:
            # Skip filters that would fail filtfilt to preserve the signal shape.
            continue
        filtered = signal.filtfilt(b, a, filtered, axis=1)
    return filtered


def moving_average(signal_array: np.ndarray, window: int) -> np.ndarray:
    """Compute a centred moving average for smoothing envelopes."""

    if window <= 1 or window >= signal_array.shape[1]:
        return signal_array
    kernel = np.ones(window, dtype=signal_array.dtype) / float(window)
    return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=1, arr=signal_array)


def zero_crossings(x: np.ndarray, threshold: float) -> np.ndarray:
    prod = x[:, :-1] * x[:, 1:]
    diff = np.abs(x[:, 1:] - x[:, :-1])
    crossings = (prod < 0) & (diff >= threshold)
    return crossings.sum(axis=1)


def slope_sign_changes(x: np.ndarray, threshold: float) -> np.ndarray:
    x1 = x[:, :-2]
    x2 = x[:, 1:-1]
    x3 = x[:, 2:]
    cond = ((x2 - x1) * (x2 - x3) >= threshold) & (np.abs(x2 - x1) >= threshold) & (np.abs(x2 - x3) >= threshold)
    return cond.sum(axis=1)


def willison_amplitude(x: np.ndarray, threshold: float) -> np.ndarray:
    return (np.abs(np.diff(x, axis=1)) >= threshold).sum(axis=1)


def hjorth_parameters(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    first_deriv = np.diff(x, axis=1)
    second_deriv = np.diff(first_deriv, axis=1)
    var0 = np.var(x, axis=1)
    var1 = np.var(first_deriv, axis=1)
    var2 = np.var(second_deriv, axis=1)

    activity = var0
    mobility = np.sqrt(np.divide(var1, var0, out=np.zeros_like(var1), where=var0 > 0))
    complexity = np.sqrt(
        np.divide(var2, var1, out=np.zeros_like(var2), where=var1 > 0)
    )
    complexity = np.divide(complexity, mobility, out=np.zeros_like(complexity), where=mobility > 0)
    return activity, mobility, complexity


def spectral_descriptors(x: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = x.shape[1]
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    fft = np.abs(np.fft.rfft(x, axis=1)) ** 2
    power = fft + 1e-12
    total_power = power.sum(axis=1, keepdims=True)
    mean_freq = np.divide((power * freqs).sum(axis=1), total_power.squeeze(), out=np.zeros(x.shape[0]), where=total_power.squeeze() > 0)
    cumulative_power = np.cumsum(power, axis=1)
    half_power = total_power / 2
    median_freq_idx = np.array([
        np.searchsorted(cumulative_power[i], half_power[i]) for i in range(power.shape[0])
    ])
    median_freq = freqs[np.clip(median_freq_idx, 0, len(freqs) - 1)]
    power_norm = power / total_power
    entropy = -np.sum(power_norm * np.log(power_norm + 1e-12), axis=1)
    entropy = entropy / np.log(power.shape[1])
    return mean_freq, median_freq, entropy


def extract_features_from_emg(
    emg: np.ndarray,
    cfg: FeatureExtractionConfig,
) -> np.ndarray:
    """Compute a comprehensive feature vector from a multi-channel EMG snippet."""

    channels, timesteps = emg.shape
    threshold = cfg.zc_threshold_ratio * np.std(emg, axis=1, keepdims=True) + 1e-6
    threshold = threshold.squeeze()
    wamp_threshold = cfg.wamp_threshold_ratio * np.std(emg, axis=1, keepdims=True) + 1e-6
    wamp_threshold = wamp_threshold.squeeze()

    abs_emg = np.abs(emg)
    smoothed_abs = moving_average(abs_emg, int(cfg.smoothing_window_ms * cfg.sample_rate / 1000))

    diff = np.diff(emg, axis=1)

    rms = np.sqrt(np.mean(emg ** 2, axis=1))
    mav = np.mean(abs_emg, axis=1)
    wl = np.sum(np.abs(diff), axis=1)
    zc = zero_crossings(emg, threshold)
    ssc = slope_sign_changes(emg, threshold)
    wamp = willison_amplitude(emg, wamp_threshold)
    var = np.var(emg, axis=1)
    std = np.std(emg, axis=1)
    max_abs = np.max(abs_emg, axis=1)
    peak_to_peak = np.ptp(emg, axis=1)
    kurt = stats.kurtosis(emg, axis=1, fisher=False, bias=False, nan_policy="omit")
    skew = stats.skew(emg, axis=1, bias=False, nan_policy="omit")
    activity, mobility, complexity = hjorth_parameters(emg)
    mean_freq, median_freq, entropy = spectral_descriptors(emg, cfg.sample_rate)

    envelope_mean = np.mean(smoothed_abs, axis=1)
    envelope_std = np.std(smoothed_abs, axis=1)
    envelope_max = np.max(smoothed_abs, axis=1)

    per_channel_features = np.stack(
        [
            rms,
            mav,
            wl,
            zc / max(timesteps - 1, 1),
            ssc / max(timesteps - 2, 1),
            wamp / max(timesteps - 1, 1),
            var,
            std,
            max_abs,
            peak_to_peak,
            kurt,
            skew,
            activity,
            mobility,
            complexity,
            mean_freq,
            median_freq,
            entropy,
            envelope_mean,
            envelope_std,
            envelope_max,
        ],
        axis=1,
    )  # shape: (channels, n_features_per_channel)

    # Enhanced: capture cross-channel synergies using correlation of rectified signals.
    corr_matrix = np.corrcoef(abs_emg)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    upper_indices = np.triu_indices(channels, k=1)
    correlation_features = corr_matrix[upper_indices] * cfg.correlation_weight

    global_mean = np.mean(per_channel_features, axis=0)
    global_std = np.std(per_channel_features, axis=0)

    feature_vector = np.concatenate([
        per_channel_features.reshape(-1),
        global_mean,
        global_std,
        correlation_features,
    ])
    return np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)


def load_emg_from_file(path: Path) -> np.ndarray:
    """Load an EMG recording from the provided MATLAB file."""

    mat = sio.loadmat(path)
    if "data" not in mat:
        raise KeyError(f"File {path} does not contain a 'data' variable")
    data = np.asarray(mat["data"], dtype=np.float32)
    if data.ndim == 3:
        emg = data[0]
    elif data.ndim == 2:
        emg = data
    elif data.ndim == 1:
        emg = data.reshape(-1, 1)
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unexpected EMG data shape {data.shape} for {path}")
    return emg.T  # (channels, timesteps)


def prepare_split_features(
    dataset: EMGDataset15,
    limit: int,
    cfg: FeatureExtractionConfig,
    rng: np.random.Generator,
    shuffle: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert the dataset split into handcrafted features and label arrays."""

    indices = np.arange(len(dataset.file_list))
    if shuffle:
        rng.shuffle(indices)
    if limit > 0:
        indices = indices[:limit]

    filters = design_emg_filters(cfg.sample_rate)
    features: List[np.ndarray] = []
    labels: List[int] = []

    for idx, dataset_index in enumerate(indices, start=1):
        label_str, path = dataset.file_list[dataset_index]
        emg = load_emg_from_file(Path(path))
        if emg.shape[1] > cfg.trim_start:
            emg = emg[:, cfg.trim_start :]
        emg = emg - emg.mean(axis=1, keepdims=True)
        emg = apply_filters(emg, filters)
        features.append(extract_features_from_emg(emg, cfg))
        labels.append(int(label_str))

        if idx % 250 == 0:
            print(f"   Processed {idx}/{len(indices)} samples", flush=True)

    if not features:
        raise RuntimeError("No features extracted. Check dataset path and limits.")

    return np.stack(features, axis=0), np.asarray(labels, dtype=np.int64)


# ----------------------------------------------------------------------
# 🔧 Class balancing utility
# ----------------------------------------------------------------------
def balance_training_data(
    X: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform simple random over-sampling so every class matches the majority count."""

    class_counts = Counter(y.tolist())
    max_count = max(class_counts.values())
    if len(class_counts) <= 1:
        return X, y
    augmented_features = [X]
    augmented_labels = [y]

    for cls, count in class_counts.items():
        if count == max_count:
            continue
        indices = np.where(y == cls)[0]
        required = max_count - count
        sampled = rng.choice(indices, size=required, replace=True)
        augmented_features.append(X[sampled])
        augmented_labels.append(y[sampled])

    X_balanced = np.vstack(augmented_features)
    y_balanced = np.concatenate(augmented_labels)
    perm = rng.permutation(len(y_balanced))
    return X_balanced[perm], y_balanced[perm]


# ----------------------------------------------------------------------
# ✅ Benchmark function using EMGClassifier
# ----------------------------------------------------------------------
def benchmark_classifier(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int,
) -> Tuple[float, float, float]:
    """Fit a LibEMG classifier and return train/val/test accuracies."""

    params = MODEL_PARAMS.get(model_name, {}).copy()
    # Inject a deterministic random state when available.
    if "random_state" in params:
        params["random_state"] = random_state

    clf = EMGClassifier(model_name, model_parameters=params, random_seed=random_state)
    feature_dict_train = {"training_features": X_train, "training_labels": y_train}
    clf.fit(feature_dictionary=feature_dict_train)

    train_pred, _ = clf.run(X_train)
    val_pred, _ = clf.run(X_val)
    test_pred, _ = clf.run(X_test)

    return (
        accuracy_score(y_train, train_pred),
        accuracy_score(y_val, val_pred),
        accuracy_score(y_test, test_pred),
    )


# ----------------------------------------------------------------------
# ✅ Main execution logic
# ----------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rng = np.random.default_rng(args.random_state)

    print("📦 Loading EMG dataset splits...", flush=True)
    cfg = MFSCConfig()
    feature_cfg = FeatureExtractionConfig(sample_rate=cfg.sample_rate, trim_start=cfg.trim_start)

    splits = {
        "train": EMGDataset15(str(args.dataset_root), split="train", cfg=cfg),
        "val": EMGDataset15(str(args.dataset_root), split="val", cfg=cfg),
        "test": EMGDataset15(str(args.dataset_root), split="test", cfg=cfg),
    }

    limits = {"train": args.max_train, "val": args.max_val, "test": args.max_test}
    arrays: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for split, dataset in splits.items():
        print(f"\n⚙️ Extracting handcrafted features for {split} split (limit={limits[split]})...", flush=True)
        X, y = prepare_split_features(
            dataset,
            limit=limits[split],
            cfg=feature_cfg,
            rng=rng,
            shuffle=(split == "train"),
        )
        arrays[split] = (X.astype(np.float32) if args.float32 else X, y)
        print(f"✅ {split.capitalize()} features ready → {arrays[split][0].shape}")

    X_train, y_train = arrays["train"]
    X_val, y_val = arrays["val"]
    X_test, y_test = arrays["test"]

    print("\n📊 Training class distribution (before balancing):")
    for cls, count in sorted(Counter(y_train.tolist()).items()):
        print(f"   Class {cls:>2}: {count}")

    print("\n⚖️ Balancing training data via random over-sampling...", flush=True)
    X_train_bal, y_train_bal = balance_training_data(X_train, y_train, rng)
    print(f"✅ Balanced training set size: {X_train_bal.shape[0]} samples")

    print("\n⚙️ Standardising features using training statistics...", flush=True)
    scaler = StandardScaler()
    X_train_bal_scaled = scaler.fit_transform(X_train_bal)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    if args.no_pca:
        X_train_pca = X_train_scaled
        X_train_bal_pca = X_train_bal_scaled
        X_val_pca = X_val_scaled
        X_test_pca = X_test_scaled
        effective_components = X_train_scaled.shape[1]
    else:
        print("⚙️ Running PCA for dimensionality reduction (target ≥99% variance)...", flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            pca_probe = PCA(n_components=min(feature_cfg.pca_max_components, X_train_bal_scaled.shape[1]))
            pca_probe.fit(X_train_bal_scaled)
        cumulative_variance = np.cumsum(pca_probe.explained_variance_ratio_)
        n_components = int(np.searchsorted(cumulative_variance, feature_cfg.pca_variance) + 1)
        n_components = min(n_components, feature_cfg.pca_max_components, X_train_bal_scaled.shape[1])
        explained = cumulative_variance[n_components - 1]
        print(f"   → Retaining {n_components} components (captures {explained:.2%} variance)")
        pca = PCA(n_components=n_components, random_state=args.random_state)
        X_train_bal_pca = pca.fit_transform(X_train_bal_scaled)
        X_train_pca = pca.transform(X_train_scaled)
        X_val_pca = pca.transform(X_val_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        effective_components = n_components

    print(f"✅ Feature dimensionality after preprocessing: {effective_components}")

    models = ["LDA", "KNN", "SVM", "QDA", "RF", "NB", "GB", "MLP"]
    results: List[Dict[str, float]] = []

    for model_name in models:
        print(f"\n▶ Running {model_name} classifier…", flush=True)
        start = time.perf_counter()
        train_acc, val_acc, test_acc = benchmark_classifier(
            model_name,
            X_train_bal_pca,
            y_train_bal,
            X_val_pca,
            y_val,
            X_test_pca,
            y_test,
            random_state=args.random_state,
        )
        elapsed = time.perf_counter() - start
        results.append(
            {
                "Model": model_name,
                "Train Acc": train_acc * 100.0,
                "Val Acc": val_acc * 100.0,
                "Test Acc": test_acc * 100.0,
                "Time (s)": elapsed,
            }
        )
        print(
            f"{model_name}: Train={train_acc:.2%} | Val={val_acc:.2%} | Test={test_acc:.2%} | Time={elapsed:.1f}s",
            flush=True,
        )

    df = pd.DataFrame(results).sort_values("Test Acc", ascending=False)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    print("\n🏁 Summary (sorted by Test Accuracy):")
    print(df.to_string(index=False))
    print(f"\n💾 Results saved to: {args.output_csv.resolve()}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())

