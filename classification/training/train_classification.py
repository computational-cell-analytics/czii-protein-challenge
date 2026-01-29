import os

import argparse
import torch
import numpy as np
import json

from torch_em.classification.classification_logger import ClassificationLogger
from torch_em.classification.classification_trainer import ClassificationTrainer

from classification.training import get_paths, get_coords_and_targets, get_data
from classification.utils import classification_training,ClassificationMetric,ClassificationDataset
from classification.utils.training import FocalLossWithLabelSmoothing

'''
#EXPERIMENTAL DATA
EX_TRAIN_ROOT = "/scratch-grete/projects/nim00007/cryo-et/challenge-data/public_test_dataset/data/" #"/scratch-grete/projects/nim00007/cryo-et/challenge-data/train/static/"
EX_TARGET_ROOT = "/scratch-grete/projects/nim00007/cryo-et/challenge-data/public_test_dataset/ground_truth_scaled/" #"/scratch-grete/projects/nim00007/cryo-et/challenge-data/train/overlay/ExperimentRuns/"

#SYNTHETIC DATA
TRAIN_ROOT = "/scratch-grete/projects/nim00007/cryo-et/synthetic_challenge_data/static_4/"
TARGET_ROOT ="/scratch-grete/projects/nim00007/cryo-et/synthetic_challenge_data/overlay_4/ExperimentRuns/"
'''

#All data together
TRAIN_ROOT ="/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/data/"
TARGET_ROOT ="/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/ground_truth"

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

    
    #TODO make this more automatic
    max_extent=39 #TODO check what is the biggest size from czi data/ simulation #39 or was it 35??
    halo=4
    print(f"Using bounding box size (with halo {halo}): {max_extent+halo}")
    patch_shape = (max_extent+halo, max_extent+halo, max_extent+halo)
    
    focal_loss = FocalLossWithLabelSmoothing(
        num_classes=n_classes,
        gamma=2.0,
        label_smoothing=0.1,
    )

    idx_to_label = classification_training(
        name=model_name,
        train_paths=train_paths,
        val_paths=val_paths,
        test_paths=test_paths,
        max_extent=max_extent,
        target_root = TARGET_ROOT,
        patch_shape=patch_shape,
        batch_size=64,
        lr=1e-4,
        logger=ClassificationLogger,
        trainer_class=ClassificationTrainer,
        n_iterations=1.5e3,
        out_channels=n_classes,
        in_channels=in_channels,
        loss= focal_loss, #torch.nn.CrossEntropyLoss(),#focal_loss,
        metric=ClassificationMetric(),
        augmentations=get_augmentation(),
        normalization=get_normalization(), #get_normalization(), None
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

    model_name = "protein_classification_czii_v37"
    train(args.testset, model_name)
    #mixed_train(args.testset, model_name)


if __name__ == "__main__":
    main()
