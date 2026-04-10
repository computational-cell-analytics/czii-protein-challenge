import os

import argparse
import torch
import json

from torch_em.classification.classification_logger import ClassificationLogger

from classification.utils.training import ProteinClassificationTrainer
from classification.training import get_paths
from classification.utils import classification_training, ClassificationMetric, ClassificationDataset
from classification.utils.training import FocalLossWithLabelSmoothing

#All data together
TRAIN_ROOT = "/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/data/"
TARGET_ROOT = "/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/ground_truth"

OUTPUT_ROOT = "/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/training"


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


def train(testset=True, model_name= "protein_classification"):
    in_channels=1
    n_classes = 7
    datasets = ["ExperimentRuns_faket_snr_0_12_0_2", "ExperimentRuns"]
    #model_name = "protein_classification_czii_v11"

    output_path = os.path.join(OUTPUT_ROOT, model_name)
    os.makedirs(output_path, exist_ok=True)

    train_paths = get_paths("train", datasets, TRAIN_ROOT, output_path, testset=testset)
    val_paths = get_paths("val", datasets, TRAIN_ROOT, output_path, testset=testset)
    test_paths = get_paths("test", datasets, TRAIN_ROOT, output_path, testset=testset) if testset else []

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

    halo=4 #TODO should I keep it at 4 or make it flexible and proportional to max_extent
    print(f"Using bounding box size (with halo {halo}): {max_extent+halo}")
    patch_shape = (max_extent+halo, max_extent+halo, max_extent+halo)
    
    focal_loss = FocalLossWithLabelSmoothing(
        num_classes=n_classes,
        gamma=2.0,
        label_smoothing=0.1,
    )

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
        n_iterations=100, #1.5e-3
        out_channels=n_classes,
        in_channels=in_channels,
        loss=focal_loss, #torch.nn.CrossEntropyLoss(),#focal_loss,
        metric=ClassificationMetric(),
        augmentations=get_augmentation(),
        normalization=get_normalization(), #get_normalization(), None
        save_root="/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/models",
        dataset_class=ClassificationDataset,
        num_workers=8, #TODO maybe can go bigger here
    )

    mapping_file = os.path.join(output_path, "idx_to_label.json")
    with open(mapping_file, "w") as f:
        json.dump(idx_to_label, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--testset", action='store_false', help="Set to False if no testset should be created")
    args = parser.parse_args()

    model_name = "protein_classification_czii_v41"
    train(args.testset, model_name)


if __name__ == "__main__":
    main()
