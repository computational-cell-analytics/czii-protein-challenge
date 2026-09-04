"""Stage 1 of the noise-resilience experiment: contrastive pretraining of the encoder.

Two independently corrupted views of the same subtomogram are pulled together in embedding
space. Fine-tune the result with ``train_classification.py --pretrained-encoder <ckpt dir>``.

Example:
    python pretrain_contrastive.py -n contrastive_v1
    python pretrain_contrastive.py -n contrastive_crossSNR \
        --datasets ExperimentRuns_faket_snr_0_12_0_2 \
        --paired-datasets ExperimentRuns_basic_snr_0_1_0_15 ExperimentRuns_faket_snr_1_2
"""

import argparse
import json
import os

from classification.training import get_paths, require_free_checkpoint, reuse_split
from classification.utils.training import contrastive_pretraining, CryoETNormalize, NoiseAwareViewAugment

TRAIN_ROOT = "/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/data/"
TARGET_ROOT = "/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/ground_truth/structure_for_detection/"
OUTPUT_ROOT = "/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/training"
SAVE_ROOT = "/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/models"

# Only the pick coordinates are needed, not the labels.
DATASETS = ["ExperimentRuns_faket_dens1_5_distr_eqCl3"]
N_TOMOGRAMS = None


def get_normalization():
    return CryoETNormalize(eps=1e-6, clip_percentile=0.01)


def resolve_max_extent(datasets):
    """Same lookup as supervised training, so the crops line up."""
    found = []
    for dataset in datasets:
        info_file = os.path.join(TRAIN_ROOT, dataset, "max_extent.json")
        if os.path.exists(info_file):
            with open(info_file) as f:
                value = json.load(f).get("max_extent")
            if value is not None:
                found.append(value)
                print(f"Found max_extent={value} in {dataset}")
    return max(found) if found else 39


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-n", "--name", default="protein_classification_contrastive_v1", help="Checkpoint name.")
    parser.add_argument("-d", "--datasets", nargs="+", default=DATASETS, help="Dataset folders to pretrain on.")
    parser.add_argument(
        "--max-extent", type=int, default=None,
        help="Override the crop size lookup. Required for datasets without a max_extent.json, so "
             "the pretraining crops match the ones the classifier is fine-tuned/run on.",
    )
    parser.add_argument("-t", "--testset", action="store_false", help="Set to False if no testset should be created.")
    parser.add_argument("-i", "--iterations", type=int, default=int(2e4), help="Number of training iterations.")
    parser.add_argument("-b", "--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.2, help="InfoNCE temperature.")
    parser.add_argument("--momentum", type=float, default=0.99, help="EMA momentum of the teacher encoder.")
    parser.add_argument("--queue-size", type=int, default=4096, help="Buffered negatives; 0 = in-batch only.")
    parser.add_argument(
        "--snr-full-db", type=float, default=3.0,
        help="Above this view SNR a positive pair counts fully.",
    )
    parser.add_argument(
        "--snr-zero-db", type=float, default=-6.0,
        help="Below this view SNR the pair is dropped from the positive loss (never pushed apart).",
    )
    parser.add_argument(
        "--curriculum-fraction", type=float, default=0.5,
        help="Fraction of training over which the corruption severity ramps from mild to full. 0 disables it.",
    )
    parser.add_argument(
        "--paired-datasets", nargs="*", default=None,
        help="Dataset folders with the same tomograms at a different SNR; enables real "
             "'same particle, different SNR' positives.",
    )
    parser.add_argument("--cross-source-prob", type=float, default=0.5,
                        help="Fraction of pairs drawn from two different --paired-datasets.")
    parser.add_argument(
        "--negatives", default="downweight", choices=["downweight", "nrcl", "supervised"],
        help="What counts as a negative. downweight: ours, destroyed views are dropped from the "
             "positive loss but never repelled. nrcl: a destroyed view of the same crop becomes an "
             "explicit negative (arXiv:2509.24311). supervised: same-class keys are positives, so "
             "negatives are never from the same class.",
    )
    parser.add_argument("--nrcl-weight", type=float, default=1.0, help="Weight of the NRCL noise-aware term.")
    parser.add_argument("--noisy-view-snr-db", type=float, default=-12.0,
                        help="SNR of the destructive third view used by --negatives nrcl.")
    parser.add_argument("--no-predictor", action="store_true", help="Disable the asymmetric prediction head.")
    parser.add_argument(
        "--split-from", default=None,
        help="Training output dir of a previous run whose split-*.json to reuse. Required to keep "
             "the baseline's test tomograms out of pretraining.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing checkpoint.")
    parser.add_argument("-c", "--check", action="store_true", help="Inspect a few batches and exit.")
    args = parser.parse_args()

    if not args.check:
        require_free_checkpoint(SAVE_ROOT, args.name, args.overwrite)

    output_path = os.path.join(OUTPUT_ROOT, args.name)
    os.makedirs(output_path, exist_ok=True)
    if args.split_from:
        reuse_split(args.split_from, output_path)

    train_paths = get_paths("train", args.datasets, TRAIN_ROOT, output_path, testset=args.testset, n_tomograms=N_TOMOGRAMS)
    val_paths = get_paths("val", args.datasets, TRAIN_ROOT, output_path, testset=args.testset, n_tomograms=N_TOMOGRAMS)
    print(f"Pretraining on {len(train_paths)} train / {len(val_paths)} val tomograms")

    paired_roots = [os.path.join(TRAIN_ROOT, d) for d in args.paired_datasets] if args.paired_datasets else None
    if paired_roots:
        print(f"Cross-SNR sources: {paired_roots}")

    max_extent = args.max_extent if args.max_extent is not None else resolve_max_extent(args.datasets)
    print(f"Using max_extent={max_extent}")

    checkpoint_folder = contrastive_pretraining(
        name=args.name,
        train_paths=train_paths,
        val_paths=val_paths,
        target_root=TARGET_ROOT,
        max_extent=max_extent,
        batch_size=args.batch_size,
        lr=args.lr,
        n_iterations=args.iterations,
        use_predictor=not args.no_predictor,
        temperature=args.temperature,
        momentum=args.momentum,
        queue_size=args.queue_size,
        full_weight_db=args.snr_full_db,
        zero_weight_db=args.snr_zero_db,
        negatives=args.negatives,
        nrcl_weight=args.nrcl_weight,
        noisy_view_snr_db=args.noisy_view_snr_db,
        curriculum_fraction=args.curriculum_fraction,
        normalization=get_normalization(),
        view_augmentation=NoiseAwareViewAugment(),
        paired_dataset_roots=paired_roots,
        cross_source_prob=args.cross_source_prob,
        num_workers=args.num_workers,
        save_root=SAVE_ROOT,
        check=args.check,
    )

    if checkpoint_folder:
        with open(os.path.join(output_path, "contrastive_checkpoint.json"), "w") as f:
            json.dump({"checkpoint_folder": checkpoint_folder}, f)
        print(f"\nFine-tune with:\n  train_classification.py --pretrained-encoder {checkpoint_folder}")


if __name__ == "__main__":
    main()
