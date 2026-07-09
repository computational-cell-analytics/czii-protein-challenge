"""Unsupervised domain adaptation for the protein *classification* model.

Adapts a classifier trained on a source domain to an (unlabeled) target domain with the
Mean-Teacher approach. The classifier works on subtomogram crops, so we need candidate
crop coordinates in the target domain. Three sources are supported (``--coords_source``):

- ``detection_json`` (default): read precomputed detection results produced by
  ``detection_scripts/run_protein_detection.py`` (files ``<name>_protein_detections.json``
  containing ``[z, y, x]`` lists) from ``--detections_root``.
- ``run_detection``: run a trained detection model on each target tomogram on the fly
  (requires ``--detection_model`` and ``--json_val_path``).
- ``annotation_json``: use existing point annotations (their class labels are ignored).

The adapted checkpoint is written to ``<save_root>/checkpoints/<name>`` and an
``idx_to_label.json`` (copied from the source model) is written next to it.
"""

import argparse
import json
import os
from typing import List, Sequence, Tuple

from classification.config import MAX_EXTENT_HALO
from classification.training import get_paths, get_volume, load_peaks, get_coords_and_targets
from classification.utils.training import CryoETNormalize
from classification.utils.training.domain_adaptation import mean_teacher_adaptation


# ---------------------------------------------------------------------------
# Defaults (override on the command line). Mirrors train_classification.py.
# ---------------------------------------------------------------------------
TRAIN_ROOT = "/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/data/"
TARGET_ROOT = "/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/ground_truth/structure_for_detection/"
OUTPUT_ROOT = "/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/training"
SAVE_ROOT = "/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/models"

# Source classification model to adapt (checkpoint dir + its training-output folder).
SOURCE_MODEL_NAME = "protein_classification_czii_v66"
SOURCE_CHECKPOINT = os.path.join(SAVE_ROOT, "checkpoints", SOURCE_MODEL_NAME)


def resolve_max_extent(datasets: List[str], default: int = 39) -> int:
    """Read the largest ``max_extent`` saved during detection (mirrors train_classification)."""
    found = []
    for dataset in datasets:
        info_file = os.path.join(TRAIN_ROOT, dataset, "max_extent.json")
        if os.path.exists(info_file):
            with open(info_file) as f:
                value = json.load(f).get("max_extent")
            if value is not None:
                found.append(value)
                print(f"Found max_extent={value} in {dataset}")
    return max(found) if found else default


def coords_from_detection_json(paths: List[str], detections_root: str) -> List[List[Tuple[int, int, int]]]:
    coords_all = []
    for path in paths:
        name = os.path.basename(path)
        json_path = os.path.join(detections_root, f"{name}_protein_detections.json")
        if not os.path.exists(json_path):
            print(f"[warn] no detection json for {name}; using no crops for this tomogram.")
            coords_all.append([])
            continue
        coords = [(int(round(z)), int(round(y)), int(round(x))) for z, y, x in load_peaks(json_path)]
        coords_all.append(coords)
    return coords_all


def coords_from_annotations(paths: List[str], target_root: str) -> List[List[Tuple[int, int, int]]]:
    # Reuse the training coordinate loader but discard the (untrusted) class labels.
    coords_all, _targets = get_coords_and_targets(paths, target_root=target_root)
    return [[(int(round(z)), int(round(y)), int(round(x))) for z, y, x in coords] for coords in coords_all]


def coords_from_run_detection(
    paths: List[str], detection_model: str, json_val_path: str
) -> List[List[Tuple[int, int, int]]]:
    from detection.utils import get_prediction_torch_em, protein_detection, parse_tiling

    tiling = parse_tiling(tile_shape=None, halo=None)
    threshold = None
    coords_all = []
    for path in paths:
        volume = get_volume(path)
        pred = get_prediction_torch_em(input_volume=volume, tiling=tiling, model_path=detection_model, verbose=True)
        detections, threshold = protein_detection(pred, json_val_path, detection_model, threshold=threshold)
        coords_all.append([(int(round(z)), int(round(y)), int(round(x))) for z, y, x in detections])
    return coords_all


def get_coords(paths: List[str], args) -> List[List[Tuple[int, int, int]]]:
    if args.coords_source == "detection_json":
        assert args.detections_root, "--detections_root is required for coords_source=detection_json"
        return coords_from_detection_json(paths, args.detections_root)
    if args.coords_source == "annotation_json":
        return coords_from_annotations(paths, args.target_root)
    if args.coords_source == "run_detection":
        assert args.detection_model and args.json_val_path, \
            "--detection_model and --json_val_path are required for coords_source=run_detection"
        return coords_from_run_detection(paths, args.detection_model, args.json_val_path)
    raise ValueError(f"Unknown coords_source: {args.coords_source}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-n", "--name", default="protein_classification_da_v1",
                        help="Name of the adapted-model checkpoint.")
    parser.add_argument("-m", "--source_checkpoint", default=SOURCE_CHECKPOINT,
                        help="Trained source classification model (torch_em checkpoint dir).")
    parser.add_argument("--source_idx_to_label",
                        default=os.path.join(OUTPUT_ROOT, SOURCE_MODEL_NAME, "idx_to_label.json"),
                        help="idx_to_label.json of the source model (defines class ordering / count).")
    parser.add_argument("--datasets", nargs="+", default=["ExperimentRuns"],
                        help="Target-domain dataset folder(s) under TRAIN_ROOT.")
    parser.add_argument("--train_root", default=TRAIN_ROOT)
    parser.add_argument("--target_root", default=TARGET_ROOT)
    parser.add_argument("--output_root", default=OUTPUT_ROOT)
    parser.add_argument("--save_root", default=SAVE_ROOT)

    parser.add_argument("--coords_source", choices=["detection_json", "run_detection", "annotation_json"],
                        default="detection_json")
    parser.add_argument("--detections_root", default=None,
                        help="Folder with <name>_protein_detections.json files (coords_source=detection_json).")
    parser.add_argument("--detection_model", default=None,
                        help="Detection checkpoint dir (coords_source=run_detection).")
    parser.add_argument("--json_val_path", default=None,
                        help="Detection validation split json for threshold gridsearch (coords_source=run_detection).")

    parser.add_argument("--confidence_threshold", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_iterations", type=int, default=int(2e3))
    parser.add_argument("--use_efficientnet", action="store_true")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    output_path = os.path.join(args.output_root, args.name)
    os.makedirs(output_path, exist_ok=True)

    # Reuse the source model's class ordering so pseudo-labels are consistent.
    with open(args.source_idx_to_label) as f:
        idx_to_label = json.load(f)
    n_classes = len(idx_to_label)
    print(f"Adapting classifier '{args.name}' with {n_classes} classes: {idx_to_label}")

    # Build target-domain train/val tomogram paths (no test split needed for adaptation).
    train_paths = get_paths("train", args.datasets, args.train_root, output_path, testset=False)
    val_paths = get_paths("val", args.datasets, args.train_root, output_path, testset=False)
    print(f"Target domain: {len(train_paths)} train / {len(val_paths)} val tomograms")

    max_extent = resolve_max_extent(args.datasets)
    print(f"Using max_extent={max_extent} (crop bbox {max_extent + MAX_EXTENT_HALO})")

    train_coords = get_coords(train_paths, args)
    val_coords = get_coords(val_paths, args)
    print(f"Collected {sum(len(c) for c in train_coords)} train / "
          f"{sum(len(c) for c in val_coords)} val candidate coordinates")

    normalization = CryoETNormalize(eps=1e-6, clip_percentile=0.01)

    mean_teacher_adaptation(
        name=args.name,
        source_checkpoint=args.source_checkpoint,
        train_paths=train_paths,
        val_paths=val_paths,
        train_coords=train_coords,
        val_coords=val_coords,
        max_extent=max_extent,
        n_classes=n_classes,
        in_channels=1,
        normalization=normalization,
        confidence_threshold=args.confidence_threshold,
        batch_size=args.batch_size,
        lr=args.lr,
        n_iterations=args.n_iterations,
        use_efficientnet=args.use_efficientnet,
        save_root=args.save_root,
        num_workers=args.num_workers,
        check=args.check,
    )

    # Persist the class mapping next to the adapted model for inference.
    with open(os.path.join(output_path, "idx_to_label.json"), "w") as f:
        json.dump(idx_to_label, f)
    print(f"Saved idx_to_label.json to {output_path}")


if __name__ == "__main__":
    main()
