import os

import argparse
import torch
import numpy as np
import json

from torch_em.classification.classification_logger import ClassificationLogger
from torch_em.classification.classification_trainer import ClassificationTrainer

from classification.training import get_paths, get_coords_and_targets, get_data
from classification.utils import classification_training,ClassificationMetric,ClassificationDataset

TRAIN_ROOT = "/scratch-grete/projects/nim00007/cryo-et/challenge-data/train/static/"
DETECTION_ROOT = "/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/detections/protein_detection_czii_v4/for_classification/protein_detection_czii_v4/"
TARGET_ROOT ="/scratch-grete/projects/nim00007/cryo-et/challenge-data/train/overlay/ExperimentRuns/"
OUTPUT_ROOT = "/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/training"


def get_augmentation():
    from torch_em.transform.augmentation import get_augmentations
    return get_augmentations(ndim=3)

def get_normalization():
    from torch_em.transform.raw import normalize
    return normalize

def train(testset=True):
    in_channels=1
    n_classes = 6
    datasets = ["ExperimentRuns"]
    model_name = "protein_classification_czii_v6"

    output_path = os.path.join(OUTPUT_ROOT, model_name)
    os.makedirs(output_path, exist_ok=True)

    train_paths = get_paths("train", datasets, TRAIN_ROOT, output_path, testset=testset)
    val_paths = get_paths("val", datasets, TRAIN_ROOT, output_path, testset=testset)
    test_paths = get_paths("test", datasets, TRAIN_ROOT, output_path, testset=testset) if testset else []

    train_coords, train_target = get_coords_and_targets(train_paths, target_root=TARGET_ROOT)
    val_coords, val_target = get_coords_and_targets(val_paths, target_root=TARGET_ROOT)
    test_coords, test_target = get_coords_and_targets(test_paths, target_root=TARGET_ROOT) if testset else (None, None)

    max_extent=39 #TODO check what is the biggest size from czi data/ simulation #39

    print(f"max_extent {max_extent}")

    # Now extract subtomograms
    #TODO can I include the augmentation with the coordinate being slightly off in get_data???
    train_data, train_target = get_data(train_paths, train_coords, max_extent, in_channels=in_channels, targets=train_target)
    val_data, val_target = get_data(val_paths, val_coords, max_extent, in_channels=in_channels, targets = val_target)
    test_data, test_target = get_data(test_paths, test_coords, max_extent, in_channels=in_channels, targets=test_target) if testset else None

    #TODO make this more automatic
    halo=4
    patch_shape = (64, 64, 64)
    

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
        augmentations=None, #get_augmentation(),
        normalization=None, #get_normalization(),
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
    train(args.testset)


if __name__ == "__main__":
    main()
