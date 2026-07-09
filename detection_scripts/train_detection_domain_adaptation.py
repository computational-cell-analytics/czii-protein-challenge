"""Domain adaptation for the protein *detection* model.

Adapts a detection model trained on a source domain (e.g. synthetic tomograms) to a
target domain (e.g. real experimental tomograms) using the Mean-Teacher approach.

Two modes:
- **unsupervised** (default): only the raw target tomograms are used.
- **semi-supervised**: pass ``--label_root`` to additionally use the target-domain
  labels as a supervised anchor (the same tomograms feed both the unsupervised
  consistency stream and the supervised stream, following SynapseNet's pattern).

Example
-------
    python detection_scripts/train_detection_domain_adaptation.py \
        -n protein_detection_da_v1 \
        -i /path/to/target/data/ExperimentRuns \
        -m /path/to/models/checkpoints/protein_detection_czii_v36 \
        --label_root /path/to/ground_truth/structure_for_detection/ExperimentRuns

The adapted checkpoint is written to ``<save_root>/checkpoints/<name>`` and can then be
used with ``run_protein_detection.py`` exactly like a normally trained model.
"""

import argparse
import os

from sklearn.model_selection import train_test_split

from detection.utils.training.domain_adaptation import mean_teacher_adaptation

# Same file-resolution helper used by the supervised training script.
from train_detection import find_tomogram


# ---------------------------------------------------------------------------
# Defaults (override on the command line).
# ---------------------------------------------------------------------------
TARGET_ROOT = "/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/data/ExperimentRuns"
LABEL_ROOT = "/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/ground_truth/structure_for_detection/ExperimentRuns"
SOURCE_CHECKPOINT = "/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/models/checkpoints/protein_detection_czii_v36"
SAVE_ROOT = "/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/models"


def _list_tomogram_names(target_root):
    """Names of the per-tomogram sub-directories (ignoring stray files like max_extent.json)."""
    return sorted(
        name for name in os.listdir(target_root)
        if os.path.isdir(os.path.join(target_root, name))
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-n", "--name", default="protein_detection_da_v1",
                        help="Name of the adapted-model checkpoint.")
    parser.add_argument("-i", "--input", default=TARGET_ROOT,
                        help="Folder with the target-domain tomograms (one sub-directory per tomogram).")
    parser.add_argument("-m", "--source_checkpoint", default=SOURCE_CHECKPOINT,
                        help="Path to the trained source detection model (torch_em checkpoint directory).")
    parser.add_argument("--label_root", default=None,
                        help="Folder with target-domain labels (one sub-directory per tomogram, matching --input). "
                             "If given, runs semi-supervised adaptation; otherwise unsupervised.")
    parser.add_argument("--save_root", default=SAVE_ROOT, help="Root folder for the checkpoint and logs.")
    parser.add_argument("--patch_shape", nargs=3, type=int, default=[128, 256, 256],
                        help="Training patch shape (z y x).")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_iterations", type=int, default=int(1e3))
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--confidence_threshold", type=float, default=None,
                        help="Optional confidence threshold for masking pseudo-labels. Left unset by default, "
                             "which is the appropriate choice for the regression heatmap + flow output.")
    parser.add_argument("--n_samples_train", type=int, default=None)
    parser.add_argument("--n_samples_val", type=int, default=None)
    parser.add_argument("--check", action="store_true",
                        help="Visualize the data loaders instead of running training.")
    args = parser.parse_args()

    names = _list_tomogram_names(args.input)
    if len(names) < 2:
        raise ValueError(f"Found only {len(names)} tomogram(s) in {args.input}; need at least 2 for a train/val split.")

    # Resolve raw data paths (zarr/mrc/h5/tif) and, for semi-supervised, the matching label folders.
    raw_paths = [find_tomogram(os.path.join(args.input, name)) for name in names]
    semisupervised = args.label_root is not None
    if semisupervised:
        label_paths = [os.path.join(args.label_root, name) for name in names]
        missing = [p for p in label_paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"Missing label folders: {missing}")

    # Train/val split (same split used for the unsupervised and supervised streams).
    indices = list(range(len(names)))
    train_idx, val_idx = train_test_split(indices, test_size=args.val_fraction, shuffle=True, random_state=42)
    train_paths = [raw_paths[i] for i in train_idx]
    val_paths = [raw_paths[i] for i in val_idx]

    # Zarr stores the volume at multiscale level "0"; other formats let the loader auto-derive the key.
    raw_key = "0" if all(p.endswith(".zarr") for p in raw_paths) else None

    mode = "semi-supervised" if semisupervised else "unsupervised"
    print(f"Adapting model '{args.name}' ({mode}) from source checkpoint: {args.source_checkpoint}")
    print(f"Target domain: {len(train_paths)} train / {len(val_paths)} val tomograms from {args.input}")

    supervised_kwargs = {}
    if semisupervised:
        supervised_kwargs = dict(
            supervised_train_paths=train_paths,
            supervised_val_paths=val_paths,
            supervised_train_label_paths=[label_paths[i] for i in train_idx],
            supervised_val_label_paths=[label_paths[i] for i in val_idx],
        )
        print(f"Using labels from {args.label_root} for the supervised stream.")

    mean_teacher_adaptation(
        name=args.name,
        unsupervised_train_paths=train_paths,
        unsupervised_val_paths=val_paths,
        patch_shape=tuple(args.patch_shape),
        source_checkpoint=args.source_checkpoint,
        save_root=args.save_root,
        confidence_threshold=args.confidence_threshold,
        raw_key=raw_key,
        batch_size=args.batch_size,
        lr=args.lr,
        n_iterations=args.n_iterations,
        n_samples_train=args.n_samples_train,
        n_samples_val=args.n_samples_val,
        check=args.check,
        **supervised_kwargs,
    )


if __name__ == "__main__":
    main()
