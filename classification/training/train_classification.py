import os

import argparse
import torch
import numpy as np

from torch_em.classification.classification_logger import ClassificationLogger
from torch_em.classification.classification_trainer import ClassificationTrainer

from classification.training import get_paths,get_coords_and_heatmaps, compute_max_extent_from_all, get_data
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



def train(testset=True):
    in_channels=1
    n_classes = 6
    datasets = ["ExperimentRuns"]
    model_name = "protein_classification_czii_v1"

    output_path = os.path.join(OUTPUT_ROOT, model_name)
    os.makedirs(output_path, exist_ok=True)

    train_paths = get_paths("train", datasets, TRAIN_ROOT, output_path, testset=testset)
    val_paths = get_paths("val", datasets, TRAIN_ROOT, output_path, testset=testset)
    test_paths = get_paths("test", datasets, TRAIN_ROOT, output_path, testset=testset) if testset else []

    train_coords, train_heatmaps = get_coords_and_heatmaps(train_paths, detection_root=DETECTION_ROOT)
    val_coords, val_heatmaps = get_coords_and_heatmaps(val_paths, detection_root=DETECTION_ROOT)
    test_coords, test_heatmaps = get_coords_and_heatmaps(test_paths, detection_root=DETECTION_ROOT) if testset else ([], [])
    
    # Compute max_extent
    all_peaks = train_coords + val_coords + test_coords
    all_heatmaps = train_heatmaps + val_heatmaps + test_heatmaps
    max_extent = compute_max_extent_from_all(all_peaks, all_heatmaps)

    print(f"max_extent {max_extent}")

    # Now extract subtomograms
    #TODO can I include the augmentation with the coordinate being slightly off in get_data???
    train_data = get_data(train_paths, train_coords, max_extent, in_channels=in_channels)
    val_data = get_data(val_paths, val_coords, max_extent, in_channels=in_channels)
    test_data = get_data(test_paths, test_coords, max_extent, in_channels=in_channels) if testset else None

    patch_shape = train_data[0].shape
    

    classification_training(
        name=model_name,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        patch_shape=patch_shape,
        batch_size=2,
        lr=1e-4,
        logger=ClassificationLogger,
        trainer_class=ClassificationTrainer,
        n_iterations=1e5,
        out_channels=n_classes,
        in_channels=in_channels,
        loss=torch.nn.CrossEntropyLoss(),
        metric=ClassificationMetric(),
        augmentations=None,
        normalization=None,
        save_root="/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/models",
        dataset_class=ClassificationDataset,
    )



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--testset", action='store_false', help="Set to False if no testset should be created")
    args = parser.parse_args()
    train(args.testset)


if __name__ == "__main__":
    main()
