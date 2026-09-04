import os

import argparse
import torch
import json

from torch_em.classification.classification_logger import ClassificationLogger

from classification.utils.training import ProteinClassificationTrainer
from classification.training import get_paths, require_free_checkpoint, reuse_split
from classification.utils import classification_training, ClassificationMetric, ClassificationDataset
from classification.utils.training import FocalLossWithLabelSmoothing, BalancedSoftmaxLoss
from classification.config import MAX_EXTENT_HALO

#All data together
TRAIN_ROOT = "/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/data/"
TARGET_ROOT = "/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/ground_truth/structure_for_detection/" #"/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/ground_truth"

OUTPUT_ROOT = "/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/training"
SAVE_ROOT = "/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/models"


def get_augmentation():
    '''from torch_em.transform.augmentation import get_augmentations
    return get_augmentations(ndim=3)'''

    from classification.utils.training import CryoETAugment
    return CryoETAugment()


def get_normalization():
    '''from torch_em.transform.raw import normalize
    return normalize'''

    from classification.utils.training import CryoETNormalize
    return CryoETNormalize(
        eps=1e-6,
        clip_percentile=0.01,
    )


def build_loss(loss_name, n_classes):
    """Construct the training loss from a short name.

    "focal"    -> FocalLossWithLabelSmoothing (inverse-frequency alpha, sqrt-tempered)
    "balanced_softmax" -> BalancedSoftmaxLoss (logit adjustment, tau=1.0)
    "ce"       -> plain CrossEntropyLoss
    """
    if loss_name == "focal":
        return FocalLossWithLabelSmoothing(
            num_classes=n_classes,
            gamma=2.0,
            alpha="balanced",  # inverse-frequency weights, computed from the training data
            alpha_beta=0.3,    # sqrt-tempered: gentler than full inverse-frequency (beta=1)
            label_smoothing=0.1,
        )
    if loss_name == "balanced_softmax":
        # Class-frequency prior is filled in from the training targets at runtime.
        # tau=1.0 is standard Balanced Softmax; lower tau tempers the correction.
        return BalancedSoftmaxLoss(
            num_classes=n_classes,
            tau=1.0,
            prior="balanced",
            label_smoothing=0.1,
        )
    if loss_name == "ce":
        return torch.nn.CrossEntropyLoss()
    raise ValueError(f"Unknown loss '{loss_name}'. Choose from: focal, balanced_softmax, ce.")


def train(
    testset=True,
    model_name="protein_classification",
    loss_name="focal",
    pretrained_encoder=None,
    pretrained_use_teacher=False,
    encoder_lr_scale=1.0,
    split_from=None,
    overwrite=False,
):
    #variables
    if model_name in (None, "protein_classification"):
        model_name = "protein_classification_czii_v80"
    in_channels = 1
    n_classes = 7
    datasets = ["ExperimentRuns_faket_dens1_5_distr_eqCl3", "ExperimentRuns_basicNoise_dens1_5_distr_eqCl3"] 
    # Limit tomograms per dataset. Set to None to use all, a single int for a uniform
    # limit, or a dict for per-dataset control, e.g.:
    # N_TOMOGRAMS = {"ExperimentRuns_faket_dens1_5_distr_eqCl2": 5, "ExperimentRuns_basicNoise_dens1_5_distr_eqCl2": 3} or
    N_TOMOGRAMS = {"ExperimentRuns_faket_dens1_5_distr_eqCl3": 25, "ExperimentRuns_basicNoise_dens1_5_distr_eqCl3": 25} 
    #N_TOMOGRAMS = {"ExperimentRuns_faket_dens1_5_distr_eqCl3": 25, "ExperimentRuns_faket_dens1_5_distr_eqCl2": 25}


    require_free_checkpoint(SAVE_ROOT, model_name, overwrite)

    output_path = os.path.join(OUTPUT_ROOT, model_name)
    os.makedirs(output_path, exist_ok=True)
    if split_from:
        reuse_split(split_from, output_path)

    train_paths = get_paths("train", datasets, TRAIN_ROOT, output_path, testset=testset, n_tomograms=N_TOMOGRAMS)
    val_paths = get_paths("val", datasets, TRAIN_ROOT, output_path, testset=testset, n_tomograms=N_TOMOGRAMS)
    test_paths = get_paths("test", datasets, TRAIN_ROOT, output_path, testset=testset, n_tomograms=N_TOMOGRAMS) if testset else []

    print(f"Using {len(train_paths)} tomograms as train set")
    print(f"Using {len(val_paths)} tomograms as val set")
    print(f"Using {len(test_paths)} tomograms as test set")

    # Default value if no max_extent.json is found
    max_extent = 39
    found_values = []

    #search for saved information about max_extent (saved in detection part)
    #if there are multiple max_extent.json, keep the largest
    for dataset in datasets:
        dataset_folder = os.path.join(TRAIN_ROOT, dataset)
        info_file_path = os.path.join(dataset_folder, "max_extent.json")

        if os.path.exists(info_file_path):
            with open(info_file_path, "r") as f:
                data = json.load(f)
            value = data.get("max_extent")
            if value is not None:
                found_values.append(value)
                print(f"Found max_extent={value} in {dataset}")

    if found_values:
        max_extent = max(found_values)

    halo = MAX_EXTENT_HALO
    print(f"Using bounding box size (with halo {halo}): {max_extent+halo}")
    patch_shape = (max_extent+halo, max_extent+halo, max_extent+halo)
    
    loss = build_loss(loss_name, n_classes)
    print(f"Using loss: {loss_name} ({type(loss).__name__})")

    if pretrained_encoder:
        print(f"Fine-tuning from contrastive encoder: {pretrained_encoder} "
              f"({'teacher' if pretrained_use_teacher else 'online'} weights, lr scale {encoder_lr_scale})")

    print(f"Training model {model_name}")

    idx_to_label = classification_training(
        name=model_name,
        train_paths=train_paths,
        val_paths=val_paths,
        test_paths=test_paths,
        max_extent=max_extent,
        target_root=TARGET_ROOT,
        patch_shape=patch_shape,
        batch_size=64,
        lr=1e-4,
        logger=ClassificationLogger,
        trainer_class=ProteinClassificationTrainer,
        n_iterations=8e3,
        out_channels=n_classes,
        in_channels=in_channels,
        loss=loss,
        metric=ClassificationMetric(),
        augmentations=get_augmentation(),
        normalization=get_normalization(), #get_normalization(), None
        save_root=SAVE_ROOT,
        dataset_class=ClassificationDataset,
        num_workers=8, #TODO maybe can go bigger here
        pretrained_encoder=pretrained_encoder,
        pretrained_use_teacher=pretrained_use_teacher,
        encoder_lr_scale=encoder_lr_scale,
    )

    mapping_file = os.path.join(output_path, "idx_to_label.json")
    with open(mapping_file, "w") as f:
        json.dump(idx_to_label, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--testset", action='store_false', help="Set to False if no testset should be created")
    parser.add_argument(
        "-l", "--loss", default="focal", choices=["focal", "balanced_softmax", "ce"],
        help="Training loss: focal (default), balanced_softmax, or ce.",
    )
    parser.add_argument("-n", "--name", default=None, help="Checkpoint name (defaults to the one set in train()).")
    parser.add_argument(
        "-p", "--pretrained-encoder", default=None,
        help="Checkpoint folder of a contrastively pretrained encoder (see pretrain_contrastive.py).",
    )
    parser.add_argument(
        "--pretrained-use-teacher", action="store_true",
        help="Initialize from the EMA teacher instead of the online encoder.",
    )
    parser.add_argument(
        "--encoder-lr-scale", type=float, default=1.0,
        help="LR multiplier for the pretrained backbone relative to the classification head.",
    )
    parser.add_argument(
        "--split-from", default=None,
        help="Training output dir of a previous run whose split-*.json to reuse, so runs are "
             "compared on identical train/val/test tomograms.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing checkpoint.")
    args = parser.parse_args()

    train(
        args.testset,
        model_name=args.name,
        loss_name=args.loss,
        pretrained_encoder=args.pretrained_encoder,
        pretrained_use_teacher=args.pretrained_use_teacher,
        encoder_lr_scale=args.encoder_lr_scale,
        split_from=args.split_from,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
