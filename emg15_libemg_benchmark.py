"""Benchmark LibEMG EMGClassifier on the 15-channel EMG dataset."""
from __future__ import annotations

import argparse
import inspect
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score

# ----------------------------------------------------------------------
# ✅ Path and CRC Fix (Option 1)
# ----------------------------------------------------------------------
import types

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
except Exception as e:
    print(f"⚠️ CRC patch warning: {e}")

# ----------------------------------------------------------------------
# ✅ Import EMGClassifier directly from libemg.emg_predictor
# ----------------------------------------------------------------------
try:
    from libemg.emg_predictor import EMGClassifier
except ImportError as exc:
    raise SystemExit(
        f"❌ Could not import EMGClassifier from libemg.emg_predictor: {exc}\n"
        "Make sure you run this script from inside the 'libemg-main' directory."
    )

# ----------------------------------------------------------------------
# ✅ Import your dataset config
# ----------------------------------------------------------------------
from emg15_dataset import EMGDataset15, MFSCConfig

# ----------------------------------------------------------------------
# ✅ CLI argument parsing
# ----------------------------------------------------------------------
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=Path("runs/15vc/libemg_benchmark_results.csv"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--float32", action="store_true")
    return parser.parse_args(argv)

# ----------------------------------------------------------------------
# ✅ Dataset helper functions
# ----------------------------------------------------------------------
def dataset_to_numpy(dataset: EMGDataset15, *, batch_size: int, num_workers: int) -> Tuple[np.ndarray, np.ndarray]:
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    X, y = [], []
    for features, labels in loader:
        X.append(features.cpu().numpy())
        y.append(labels.cpu().numpy())
    return np.concatenate(X, axis=0), np.concatenate(y, axis=0)

def flatten_features(X: np.ndarray, as_float32: bool) -> np.ndarray:
    Xf = X.reshape(X.shape[0], -1)
    return Xf.astype(np.float32, copy=False) if as_float32 else Xf

# ----------------------------------------------------------------------
# ✅ Benchmark function using EMGClassifier
# ----------------------------------------------------------------------
def benchmark_classifier(model_name: str, X_train, y_train, X_val, y_val, X_test, y_test):
    clf = EMGClassifier(model_name)
    feature_dict_train = {"training_features": X_train, "training_labels": y_train}
    clf.fit(feature_dictionary=feature_dict_train)

    train_pred, _ = clf.run(X_train)
    val_pred, _ = clf.run(X_val)
    test_pred, _ = clf.run(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    test_acc = accuracy_score(y_test, test_pred)
    return train_acc, val_acc, test_acc

# ----------------------------------------------------------------------
# ✅ Main function
# ----------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)

    splits = {
        "train": EMGDataset15(str(args.dataset_root), split="train"),
        "val": EMGDataset15(str(args.dataset_root), split="val"),
        "test": EMGDataset15(str(args.dataset_root), split="test"),
    }

    arrays = {}
    for split, dataset in splits.items():
        print(f"Preparing {split} split with {len(dataset)} samples …", flush=True)
        X, y = dataset_to_numpy(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
        arrays[split] = (flatten_features(X, as_float32=args.float32), y)

    X_train, y_train = arrays["train"]
    X_val, y_val = arrays["val"]
    X_test, y_test = arrays["test"]

    models = ["LDA", "KNN", "SVM", "QDA", "RF", "NB", "GB", "MLP"]
    results = []

    for model_name in models:
        print(f"\nRunning {model_name} classifier …", flush=True)
        start = time.perf_counter()
        train_acc, val_acc, test_acc = benchmark_classifier(model_name, X_train, y_train, X_val, y_val, X_test, y_test)
        elapsed = time.perf_counter() - start
        results.append({
            "Model": model_name,
            "Train Acc": train_acc * 100,
            "Val Acc": val_acc * 100,
            "Test Acc": test_acc * 100,
            "Time (s)": elapsed,
        })
        print(f"{model_name}: Train={train_acc:.3%}, Val={val_acc:.3%}, Test={test_acc:.3%}, Time={elapsed:.2f}s")

    import pandas as pd
    df = pd.DataFrame(results).sort_values("Test Acc", ascending=False)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print("\n✅ Summary (sorted by Test Accuracy):")
    print(df)
    print(f"\nResults saved to: {args.output_csv.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
