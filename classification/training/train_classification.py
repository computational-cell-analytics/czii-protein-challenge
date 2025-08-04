import os
import sys
sys.path.append("/user/muth9/u12095/czii-protein-challenge")

import argparse
import torch
import numpy as np

from data_utils import get_data
from torch_em.classification.classification_logger import ClassificationLogger
from torch_em.classification.classification_trainer import ClassificationTrainer
from classification.utils import classification_training,ClassificationMetric,ClassificationDataset

TRAIN_ROOT = "/scratch-grete/projects/nim00007/cryo-et/challenge-data/train/static/"
DETECTION_ROOT = "/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/detections/protein_detection_czii_v4/for_classification/protein_detection_czii_v4/"
OUTPUT_ROOT = "/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/training"


# Dummy functions for augmentation and normalization #TODO!!!!!!!!!!
def dummy_augmentation(volume: np.ndarray) -> np.ndarray:
    return volume


def dummy_normalization(volume: np.ndarray) -> np.ndarray:
    volume = volume.astype(np.float32)
    return (volume - np.min(volume)) / (np.max(volume) - np.min(volume) + 1e-8)



def train(key, ignore_label=None, training_2D=False, testset=True, extension="zarr"):
    n_classes = 6
    datasets = ["ExperimentRuns"]
    model_name = "protein_classification_czii_v1"

    output_path = os.path.join(OUTPUT_ROOT, model_name)
    os.makedirs(output_path, exist_ok=True)

    train_data = get_data(
        "train", datasets=datasets, train_root=TRAIN_ROOT, detection_root=DETECTION_ROOT,
        output_root=output_path, n_classes=n_classes, testset=testset
    )

    val_data = get_data(
        "val", datasets=datasets, train_root=TRAIN_ROOT, detection_root=DETECTION_ROOT,
        output_root=output_path, n_classes=n_classes, testset=testset
    )

    if testset:
        test_data = get_data(
            "test", datasets=datasets, train_root=TRAIN_ROOT, detection_root=DETECTION_ROOT,
            output_root=output_path, n_classes=n_classes, testset=testset
        )
    else:
        test_data = None
    
    patch_shape = None
    batch_size = 2
    check = False

    classification_training(
        name=model_name,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        patch_shape=patch_shape,
        batch_size=batch_size,
        lr=1e-4,
        logger=ClassificationLogger,
        trainer_class=ClassificationTrainer,
        n_iterations=1e3,
        out_channels=n_classes,
        in_channels=1,
        loss=torch.nn.CrossEntropyLoss(),
        metric=ClassificationMetric(),
        augmentations=dummy_augmentation,
        normalization=dummy_normalization,
        save_root="/mnt/lustre-emmy-hdd/usr/u12095/cryo-et/czii_challenge/models",
        dataset_class=ClassificationDataset,
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--testset", action='store_false', help="Set to False if no testset should be created")
    args = parser.parse_args()
    train(args.testset)


if __name__ == "__main__":
    main()
