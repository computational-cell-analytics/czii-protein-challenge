"""Accuracy vs. injected SNR, for comparing a baseline against a pretrained checkpoint.

Only noise is injected -- no geometry, no optics -- so the sweep isolates noise resilience.

Example:
    python eval_noise_robustness.py \
        --checkpoints /path/models/checkpoints/protein_classification_czii_v71 \
                      /path/models/checkpoints/protein_classification_czii_v72_contrastive \
        --split-dir /path/training/protein_classification_czii_v71 \
        --datasets ExperimentRuns_faket_dens1_5_distr_eqCl3
"""

import argparse
import csv
import json
import os
from typing import Dict, List

import numpy as np
import torch
from sklearn import metrics

from classification.training import get_paths, require_free_file
from classification.utils.inference.protein_classification import get_model
from classification.utils.training import CryoETNormalize, FixedNoiseTransform, SingleViewDataset

TRAIN_ROOT = "/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/data/"
TARGET_ROOT = "/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/ground_truth/structure_for_detection/"

DEFAULT_SNR_DB = [None, 12.0, 9.0, 6.0, 3.0, 0.0, -3.0, -6.0, -9.0]


def load_idx_to_label(checkpoint) -> Dict[int, str]:
    idx_to_label = checkpoint.get("idx_to_label") or checkpoint.get("init", {}).get("idx_to_label")
    if idx_to_label is None:
        raise ValueError("Checkpoint has no idx_to_label; cannot map predictions to class names.")
    return {int(k): v for k, v in idx_to_label.items()}


@torch.no_grad()
def predict(model, loader, device) -> tuple:
    y_true, y_pred = [], []
    for x, y in loader:
        logits = model(x.to(device, non_blocking=True))
        y_pred.append(logits.argmax(dim=1).cpu().numpy())
        y_true.append(y.numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)


def evaluate_checkpoint(
    checkpoint_dir: str,
    paths: List[str],
    max_extent: int,
    snr_levels: List[float],
    batch_size: int,
    num_workers: int,
    correlated: bool,
    n_classes: int,
) -> List[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = get_model(checkpoint_dir, device)
    idx_to_label = load_idx_to_label(checkpoint)
    label_to_index = {label: idx for idx, label in idx_to_label.items()}
    if len(label_to_index) != n_classes:
        print(f"[warn] checkpoint declares {len(label_to_index)} classes, expected {n_classes}")

    rows = []
    for snr_db in snr_levels:
        dataset = SingleViewDataset(
            paths=paths,
            target_root=TARGET_ROOT,
            max_extent=max_extent,
            normalization=CryoETNormalize(eps=1e-6, clip_percentile=0.01),
            transform=FixedNoiseTransform(snr_db, correlated=correlated),
            label_to_index=label_to_index,
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        )
        y_true, y_pred = predict(model, loader, device)

        keep = y_true >= 0  # crops of a class the model does not know
        y_true, y_pred = y_true[keep], y_pred[keep]

        row = {
            "checkpoint": os.path.basename(os.path.normpath(checkpoint_dir)),
            "snr_db": "clean" if snr_db is None else snr_db,
            "n": int(y_true.size),
            "accuracy": float(metrics.accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(metrics.balanced_accuracy_score(y_true, y_pred)),
            "macro_f1": float(metrics.f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }
        per_class = metrics.f1_score(
            y_true, y_pred, average=None, labels=sorted(idx_to_label), zero_division=0
        )
        for idx, f1 in zip(sorted(idx_to_label), per_class):
            row[f"f1_{idx_to_label[idx]}"] = float(f1)

        print(f"  SNR {str(row['snr_db']):>6}: acc {row['accuracy']:.4f}  "
              f"bal-acc {row['balanced_accuracy']:.4f}  macro-F1 {row['macro_f1']:.4f}")
        rows.append(row)

    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Checkpoint folders (containing best.pt).")
    parser.add_argument("--split-dir", required=True, help="Training output dir holding the split-*.json files.")
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--max-extent", type=int, default=None, help="Defaults to the dataset's max_extent.json.")
    parser.add_argument("--n-classes", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--correlated-noise", action="store_true", help="Use blob noise instead of white noise.")
    parser.add_argument("--snr-db", nargs="*", type=float, default=None,
                        help="SNR levels in dB. The clean baseline is always evaluated first.")
    parser.add_argument("-o", "--output", default=None, help="CSV output path.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing CSV.")
    args = parser.parse_args()

    output = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results", "noise_robustness.csv"
    )
    require_free_file(output, args.overwrite)

    paths = get_paths(args.split, args.datasets, TRAIN_ROOT, args.split_dir, testset=(args.split == "test"))
    print(f"Evaluating on {len(paths)} tomograms ({args.split} split)")

    max_extent = args.max_extent
    if max_extent is None:
        found = []
        for dataset in args.datasets:
            info_file = os.path.join(TRAIN_ROOT, dataset, "max_extent.json")
            if os.path.exists(info_file):
                with open(info_file) as f:
                    value = json.load(f).get("max_extent")
                if value is not None:
                    found.append(value)
        max_extent = max(found) if found else 39
    print(f"Using max_extent={max_extent}")

    snr_levels = [None] + list(args.snr_db) if args.snr_db else DEFAULT_SNR_DB

    rows = []
    for checkpoint_dir in args.checkpoints:
        print(f"\n== {checkpoint_dir}")
        rows.extend(evaluate_checkpoint(
            checkpoint_dir, paths, max_extent, snr_levels,
            args.batch_size, args.num_workers, args.correlated_noise, args.n_classes,
        ))

    os.makedirs(os.path.dirname(output), exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row}, key=lambda k: (k not in ("checkpoint", "snr_db"), k))
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
