import os

import argparse
import torch
import numpy as np
import json

from torch_em.classification.classification_logger import ClassificationLogger
from torch_em.classification.classification_trainer import ClassificationTrainer

from classification.training import get_paths, get_coords_and_targets, get_data
from classification.utils import classification_training,ClassificationMetric,ClassificationDataset

#EXPERIMENTAL DATA
EX_TRAIN_ROOT = "/scratch-grete/projects/nim00007/cryo-et/challenge-data/public_test_dataset/data/" #"/scratch-grete/projects/nim00007/cryo-et/challenge-data/train/static/"
EX_TARGET_ROOT = "/scratch-grete/projects/nim00007/cryo-et/challenge-data/public_test_dataset/ground_truth_scaled/" #"/scratch-grete/projects/nim00007/cryo-et/challenge-data/train/overlay/ExperimentRuns/"

#SYNTHETIC DATA
TRAIN_ROOT = "/scratch-grete/projects/nim00007/cryo-et/synthetic_challenge_data/static_4/"
TARGET_ROOT ="/scratch-grete/projects/nim00007/cryo-et/synthetic_challenge_data/overlay_4/ExperimentRuns/"

OUTPUT_ROOT = "/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/training"

def get_augmentation():
    from torch_em.transform.augmentation import get_augmentations
    return get_augmentations(ndim=3)

def get_normalization():
    from torch_em.transform.raw import normalize
    return normalize

def train(testset=True, model_name= "protein_classification"):
    in_channels=1
    n_classes = 6
    zarr_ = True
    datasets = ["ExperimentRuns", "tomograms"]
    #model_name = "protein_classification_czii_v11"

    output_path = os.path.join(OUTPUT_ROOT, model_name)
    os.makedirs(output_path, exist_ok=True)

    train_paths = get_paths("train", datasets, EX_TRAIN_ROOT, output_path, testset=testset)
    val_paths = get_paths("val", datasets, EX_TRAIN_ROOT, output_path, testset=testset)
    test_paths = get_paths("test", datasets, EX_TRAIN_ROOT, output_path, testset=testset) if testset else []

    
    #TODO make this more automatic
    max_extent=39 #TODO check what is the biggest size from czi data/ simulation #39 or was it 35??
    halo=4
    patch_shape = (max_extent+halo, max_extent+halo, max_extent+halo)
    

    idx_to_label = classification_training(
        name=model_name,
        train_paths=train_paths,
        val_paths=val_paths,
        test_paths=test_paths,
        zarr_=zarr_,
        max_extent=max_extent,
        target_root = EX_TARGET_ROOT,
        patch_shape=patch_shape,
        batch_size=64,
        lr=1e-4,
        logger=ClassificationLogger,
        trainer_class=ClassificationTrainer,
        n_iterations=5e3,
        out_channels=n_classes,
        in_channels=in_channels,
        loss=torch.nn.CrossEntropyLoss(),
        metric=ClassificationMetric(),
        augmentations=get_augmentation(), #get_augmentation(),
        normalization=None, #get_normalization(),
        save_root="/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/models",
        dataset_class=ClassificationDataset,
    )

    mapping_file = os.path.join(output_path, "idx_to_label.json")
    with open(mapping_file, "w") as f:
        json.dump(idx_to_label, f)


def mixed_train(testset=True, model_name= "protein_classification"):
    in_channels = 1
    n_classes = 6
    datasets = ["ExperimentRuns_basic"]  # synthetic
    exp_datasets = ["ExperimentRuns"]    # experimental

    output_path = os.path.join(OUTPUT_ROOT, model_name)
    os.makedirs(output_path, exist_ok=True)

    # --- synthetic paths ---
    syn_train_paths = get_paths("train", datasets, TRAIN_ROOT, output_path, testset=testset)
    syn_val_paths   = get_paths("val", datasets, TRAIN_ROOT, output_path, testset=testset)
    syn_test_paths  = get_paths("test", datasets, TRAIN_ROOT, output_path, testset=testset) if testset else []

    # --- experimental paths ---
    exp_train_paths = get_paths("train", exp_datasets, EX_TRAIN_ROOT, output_path, testset=testset)
    exp_val_paths   = get_paths("val", exp_datasets, EX_TRAIN_ROOT, output_path, testset=testset)
    exp_test_paths  = get_paths("test", exp_datasets, EX_TRAIN_ROOT, output_path, testset=testset) if testset else []

    # --- get coords + targets ---
    syn_train_coords, syn_train_target = get_coords_and_targets(syn_train_paths, target_root=TARGET_ROOT)
    exp_train_coords, exp_train_target = get_coords_and_targets(exp_train_paths, target_root=EX_TARGET_ROOT)

    syn_val_coords, syn_val_target = get_coords_and_targets(syn_val_paths, target_root=TARGET_ROOT)
    exp_val_coords, exp_val_target = get_coords_and_targets(exp_val_paths, target_root=EX_TARGET_ROOT)

    if testset:
        syn_test_coords, syn_test_target = get_coords_and_targets(syn_test_paths, target_root=TARGET_ROOT)
        exp_test_coords, exp_test_target = get_coords_and_targets(exp_test_paths, target_root=EX_TARGET_ROOT)
    else:
        syn_test_coords = syn_test_target = exp_test_coords = exp_test_target = None

    # --- combine coords + targets ---
    train_coords = syn_train_coords + exp_train_coords
    train_target = syn_train_target + exp_train_target

    val_coords = syn_val_coords + exp_val_coords
    val_target = syn_val_target + exp_val_target

    test_coords = (syn_test_coords + exp_test_coords) if testset else None
    test_target = (syn_test_target + exp_test_target) if testset else None

    max_extent = 39
    print(f"max_extent {max_extent}")

    # --- load actual data ---
    # synthetic: zarr_ = False
    syn_train_data, syn_train_target = get_data(syn_train_paths, syn_train_coords, max_extent,
                                                in_channels=in_channels, targets=syn_train_target, zarr_=False)
    syn_val_data, syn_val_target = get_data(syn_val_paths, syn_val_coords, max_extent,
                                            in_channels=in_channels, targets=syn_val_target, zarr_=False)
    if testset:
        syn_test_data, syn_test_target = get_data(syn_test_paths, syn_test_coords, max_extent,
                                                  in_channels=in_channels, targets=syn_test_target, zarr_=False)
    else:
        syn_test_data, syn_test_target = None, None

    # experimental: zarr_ = True
    exp_train_data, exp_train_target = get_data(exp_train_paths, exp_train_coords, max_extent,
                                                in_channels=in_channels, targets=exp_train_target, zarr_=True)
    exp_val_data, exp_val_target = get_data(exp_val_paths, exp_val_coords, max_extent,
                                            in_channels=in_channels, targets=exp_val_target, zarr_=True)
    if testset:
        exp_test_data, exp_test_target = get_data(exp_test_paths, exp_test_coords, max_extent,
                                                  in_channels=in_channels, targets=exp_test_target, zarr_=True)
    else:
        exp_test_data, exp_test_target = None, None

    # --- combine actual data ---
    train_data = syn_train_data + exp_train_data
    val_data   = syn_val_data + exp_val_data
    test_data  = (syn_test_data + exp_test_data) if testset else None
    train_target = syn_train_target + exp_train_target
    val_target   = syn_val_target + exp_val_target
    test_target  = (syn_test_target + exp_test_target) if testset else None

    halo = 4
    patch_shape = (max_extent+halo, max_extent+halo, max_extent+halo)

    idx_to_label = classification_training(
        name=model_name,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        train_target=train_target,
        val_target=val_target,
        test_target=test_target,
        patch_shape=patch_shape,
        batch_size=64,
        lr=1e-4,
        logger=ClassificationLogger,
        trainer_class=ClassificationTrainer,
        n_iterations=5e3,
        out_channels=n_classes,
        in_channels=in_channels,
        loss=torch.nn.CrossEntropyLoss(),
        metric=ClassificationMetric(),
        augmentations=get_augmentation(),
        normalization=None,
        save_root="/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/models",
        dataset_class=ClassificationDataset,
    )

    mapping_file = os.path.join(output_path, "idx_to_label.json")
    with open(mapping_file, "w") as f:
        json.dump(idx_to_label, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--testset", action='store_false', help="Set to False if no testset should be created")
    args = parser.parse_args()

    model_name = "protein_classification_czii_v15"
    train(args.testset, model_name)
    #mixed_train(args.testset, model_name)


if __name__ == "__main__":
    main()
