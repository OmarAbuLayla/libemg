#!/usr/bin/env python3
"""Train and evaluate an LDA classifier on 15-channel EMG .mat recordings.

The script assembles the LibEMG offline pipeline end-to-end:

1.  Recursively discover `.mat` recordings stored under Train/Val/Test splits.
2.  Load the signals into `OfflineDataHandler` instances (one per split).
3.  Apply standard LibEMG preprocessing (band-pass + notch + z-score standardisation).
4.  Window the EMG, extract TDAR features, and normalise them with `StandardScaler`.
5.  Train an `EMGClassifier` backed by scikit-learn's Linear Discriminant Analysis.
6.  Evaluate accuracy on the held-out test split and persist metrics/artifacts.

Results (accuracies and confusion matrix) are saved inside ``results_simple`` in both
CSV and pickle form so subsequent experiments can reload the artefacts.
"""
from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from libemg.data_handler import OfflineDataHandler, RegexFilter
from libemg.emg_predictor import EMGClassifier
from libemg.feature_extractor import FeatureExtractor
from libemg.filtering import Filter
from libemg.offline_metrics import OfflineMetrics


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def _strip_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def discover_metadata(root: Path) -> Tuple[List[str], List[str], List[str]]:
    """Collect unique subject/session/class identifiers across the dataset."""

    subjects: set[str] = set()
    sessions: set[str] = set()
    classes: set[str] = set()

    for split in ("Train", "Val", "Test"):
        split_root = root / split / "EMG"
        if not split_root.is_dir():
            continue
        for mat_file in split_root.rglob("*.mat"):
            try:
                rel_parts = mat_file.relative_to(split_root).parts
            except ValueError:
                # Should not happen, but guard against unexpected symlinks.
                continue
            if len(rel_parts) < 3:
                # Expect subject/session/<label>.mat
                continue
            subj_str, sess_str = rel_parts[0], rel_parts[1]
            label = mat_file.stem

            subjects.add(_strip_prefix(subj_str, "subject_"))
            sessions.add(_strip_prefix(sess_str, "session_"))
            classes.add(label)

    if not classes:
        raise RuntimeError(
            f"Could not locate any .mat recordings under {root}. "
            "Expected <root>/<Split>/EMG/<subject>/<session>/<label>.mat"
        )

    return sorted(subjects), sorted(sessions), sorted(classes)


def build_filters(subjects: Sequence[str], sessions: Sequence[str], classes: Sequence[str]) -> List[RegexFilter]:
    """Create regex filters that expose subject/session/class metadata."""

    return [
        RegexFilter("subject_", "/", list(subjects), "subject"),
        RegexFilter("session_", "/", list(sessions), "session"),
        RegexFilter("/", ".mat", list(classes), "class"),
    ]


def load_split(
    dataset_root: Path,
    split: str,
    filters: Sequence[RegexFilter],
    mat_key: str,
) -> OfflineDataHandler:
    """Load a dataset split into an OfflineDataHandler."""

    base_dir = dataset_root / split / "EMG"
    if not base_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {base_dir}")

    odh = OfflineDataHandler()
    odh.get_data(
        base_dir.as_posix(),
        regex_filters=filters,
        metadata_fetchers=None,
        mat_key=mat_key,
    )

    if not odh.data:
        raise RuntimeError(f"No EMG files discovered for split '{split}' in {base_dir}.")

    return odh


def apply_preprocessing(
    train: OfflineDataHandler,
    val: OfflineDataHandler,
    test: OfflineDataHandler,
    sampling_rate: int,
    use_filters: bool,
    use_standardize: bool,
) -> Dict[str, np.ndarray]:
    """Apply LibEMG filtering and return the standardisation statistics."""

    if use_filters:
        pre_filter = Filter(sampling_rate)
        pre_filter.install_common_filters()
        pre_filter.filter(train)
        pre_filter.filter(val)
        pre_filter.filter(test)

    stats: Dict[str, np.ndarray] = {}

    if use_standardize:
        std_filter = Filter(sampling_rate)
        std_filter.install_filters({"name": "standardize", "data": train})
        std_filter.filter(train)
        std_filter.filter(val)
        std_filter.filter(test)

        if hasattr(std_filter, "filters"):
            for filt in std_filter.filters:
                if filt.get("name") == "standardize":
                    stats["mean"] = filt.get("mean")
                    stats["std"] = filt.get("std")
                    break

    return stats


@dataclass
class FeatureOutputs:
    windows: np.ndarray
    labels: np.ndarray
    features: np.ndarray


def extract_features_for_split(
    odh: OfflineDataHandler,
    feature_extractor: FeatureExtractor,
    feature_names: Sequence[str],
    feature_params: Dict[str, float],
    window_size: int,
    window_step: int,
    scaler: StandardScaler | None,
) -> Tuple[FeatureOutputs, StandardScaler | None]:
    """Window EMG data and extract TD features for a split."""

    windows, metadata = odh.parse_windows(window_size, window_step)
    if windows.size == 0:
        raise RuntimeError("Window extraction yielded no data. Check window parameters.")

    if "class" not in metadata:
        raise KeyError("Metadata missing 'class' entries after windowing.")

    labels = metadata["class"].astype(int).reshape(-1)

    feats = feature_extractor.extract_features(
        feature_list=feature_names,
        windows=windows,
        feature_dic=feature_params,
        array=True,
    )

    if scaler is None:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(feats)
    else:
        scaled = scaler.transform(feats)

    return FeatureOutputs(windows=windows, labels=labels, features=scaled), scaler


# ---------------------------------------------------------------------------
# Training / evaluation entry point
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path, help="Root directory of the EMG dataset")
    parser.add_argument(
        "--mat-key",
        type=str,
        default="data",
        help="Variable name inside .mat files that stores the EMG tensor.",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=1000,
        help="Sampling frequency of the EMG recordings (Hz).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=200,
        help="Number of samples per analysis window.",
    )
    parser.add_argument(
        "--window-step",
        type=int,
        default=100,
        help="Stride (in samples) between consecutive windows.",
    )
    parser.add_argument(
        "--feature-group",
        type=str,
        default="TDAR",
        help="LibEMG feature group to extract (defaults to TDAR).",
    )
    parser.add_argument(
        "--ar-order",
        type=int,
        default=4,
        help="Autoregressive model order for AR features.",
    )
    parser.add_argument(
        "--null-class",
        type=int,
        default=0,
        help="Label index representing the null/no-motion class for AER.",
    )
    parser.add_argument(
        "--disable-filtering",
        action="store_true",
        help="Skip installing the default band-pass + notch preprocessing filters.",
    )
    parser.add_argument(
        "--disable-standardize",
        action="store_true",
        help="Skip channel-wise standardisation prior to feature extraction.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory to store CSV/PKL outputs (defaults to <dataset_root>/results_simple).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    dataset_root = args.dataset_root.resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    subjects, sessions, classes = discover_metadata(dataset_root)
    filters = build_filters(subjects, sessions, classes)

    print("Loading dataset splits …", flush=True)
    splits = {
        name: load_split(dataset_root, name, filters, args.mat_key)
        for name in ("Train", "Val", "Test")
    }

    print("Applying preprocessing …", flush=True)
    standardize_stats = apply_preprocessing(
        train=splits["Train"],
        val=splits["Val"],
        test=splits["Test"],
        sampling_rate=args.sampling_rate,
        use_filters=not args.disable_filtering,
        use_standardize=not args.disable_standardize,
    )

    feature_extractor = FeatureExtractor()
    feature_groups = feature_extractor.get_feature_groups()
    if args.feature_group not in feature_groups:
        raise ValueError(
            f"Unknown feature group '{args.feature_group}'. Available options: {sorted(feature_groups)}"
        )
    feature_names = feature_groups[args.feature_group]
    feature_params = {"AR_order": args.ar_order}

    print("Extracting features …", flush=True)
    split_features: Dict[str, FeatureOutputs] = {}
    scaler: StandardScaler | None = None

    for split_name, odh in splits.items():
        outputs, scaler = extract_features_for_split(
            odh,
            feature_extractor,
            feature_names,
            feature_params,
            window_size=args.window_size,
            window_step=args.window_step,
            scaler=scaler,
        )
        split_features[split_name] = outputs

    print("Training LDA classifier …", flush=True)
    classifier = EMGClassifier("LDA")
    train_outputs = split_features["Train"]
    classifier.fit(
        feature_dictionary={
            "training_features": train_outputs.features,
            "training_labels": train_outputs.labels,
        }
    )

    metrics = OfflineMetrics()
    summary_rows = []
    prediction_cache: Dict[str, Dict[str, np.ndarray]] = {}

    for split_name, outputs in split_features.items():
        preds, probs = classifier.run(outputs.features)
        acc = float(np.mean(preds == outputs.labels)) if outputs.labels.size else float("nan")
        summary_rows.append({
            "split": split_name.lower(),
            "num_windows": int(outputs.features.shape[0]),
            "accuracy": acc,
        })
        prediction_cache[split_name] = {
            "predictions": preds,
            "probabilities": probs,
            "labels": outputs.labels,
        }

    test_outputs = split_features["Test"]
    test_preds = prediction_cache["Test"]["predictions"]
    test_metrics = metrics.extract_offline_metrics(
        ["CA", "CONF_MAT", "PREC", "RECALL", "F1"],
        y_true=test_outputs.labels,
        y_predictions=test_preds,
        null_label=args.null_class,
    )

    results_dir = (
        args.results_dir.resolve()
        if args.results_dir is not None
        else (dataset_root / "results_simple")
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame(summary_rows)
    metrics_df.sort_values("split", inplace=True)
    metrics_path = results_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    conf_mat = test_metrics.get("CONF_MAT")
    conf_path = results_dir / "confusion_matrix.csv"
    if isinstance(conf_mat, np.ndarray):
        present_classes = np.sort(np.unique(test_outputs.labels))
        label_names = [classes[int(idx)] for idx in present_classes]
        conf_df = pd.DataFrame(conf_mat, index=label_names, columns=label_names)
        conf_df.to_csv(conf_path)

    pickle_path = results_dir / "run_summary.pkl"
    summary_payload = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "class_labels": classes,
        "feature_group": args.feature_group,
        "feature_names": feature_names,
        "scaler": scaler,
        "standardize_stats": standardize_stats,
        "split_metrics": summary_rows,
        "test_metrics": test_metrics,
        "predictions": prediction_cache,
    }
    with pickle_path.open("wb") as f:
        pickle.dump(summary_payload, f)

    print(f"Saved metrics to {metrics_path}")
    print(f"Saved confusion matrix to {conf_path}")
    print(f"Saved run summary to {pickle_path}")

    print("\nTest accuracy: {:.2f}%".format(test_metrics["CA"] * 100))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

