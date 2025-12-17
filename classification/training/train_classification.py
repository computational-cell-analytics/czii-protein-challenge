import os

import argparse
import torch
import numpy as np
import json

from torch_em.classification.classification_logger import ClassificationLogger
from torch_em.classification.classification_trainer import ClassificationTrainer

from classification.training import get_paths, get_coords_and_targets, get_data
from classification.utils import classification_training,ClassificationMetric,ClassificationDataset
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

class FocalLossWithLabelSmoothing(torch.nn.Module):
    """
    Combines Focal Loss with Label Smoothing for multi-class classification.
    Args:
        num_classes: number of classes
        gamma: focal loss focusing parameter (default 2.0)
        alpha: class weighting factor (float or tensor)
        label_smoothing: smoothing value epsilon (0 = no smoothing)
    """

    def __init__(self, num_classes, gamma=2.0, alpha=None, label_smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.label_smoothing = label_smoothing

        if alpha is None:
            self.alpha = torch.ones(num_classes) / num_classes
        else:
            self.alpha = torch.tensor(alpha)

    def forward(self, logits, target):
        """
        logits: (N, C)
        target: (N,) integer class labels
        """
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1) 

        #Label smoothing
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.label_smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1 - self.label_smoothing)

        #Focal loss
        probs = log_probs.exp()               # p = softmax(x)
        focal_weight = (1 - probs) ** self.gamma

        # class-weighting factor alpha
        alpha_factor = self.alpha.to(logits.device)[target]
        alpha_factor = alpha_factor.unsqueeze(1)  # shape (N,1)

        # combine all terms
        loss = -true_dist * alpha_factor * focal_weight * log_probs

        return loss.sum(dim=1).mean()

def get_augmentation():
    '''from torch_em.transform.augmentation import get_augmentations
    return get_augmentations(ndim=3)'''

    from classification.utils.training import CryoETAugment
    return CryoETAugment()

def get_normalization():
    from torch_em.transform.raw import normalize
    return normalize

def train(testset=True, model_name= "protein_classification"):
    in_channels=1
    n_classes = 7
    datasets = ["ExperimentRuns_faket_snr_1_2"]
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
        augmentations=get_augmentation(), #get_augmentation(),
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

    model_name = "protein_classification_czii_v28"
    train(args.testset, model_name)
    #mixed_train(args.testset, model_name)


if __name__ == "__main__":
    main()
