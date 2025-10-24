import os
# from glob import glob
import argparse

from detection.utils import get_paths  # noqa
from detection.utils import supervised_training  # noqa

TRAIN_ROOT = "/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/data/" #"/scratch-grete/projects/nim00007/cryo-et/challenge-data/public_test_dataset/data/" #"/mnt/vast-nhr/home/muth9/u12095/cryo-et/czii_challenge/data/raw" #"/scratch-grete/projects/nim00007/cryo-et/challenge-data/public_test_dataset/data/" #"/scratch-grete/projects/nim00007/cryo-et/challenge-data/train/static/"
LABEL_ROOT = "/scratch-grete/projects/nim00007/cryo-et/challenge-data/public_test_dataset/ground_truth_scaled_for_detection/" #"/mnt/vast-nhr/home/muth9/u12095/cryo-et/czii_challenge/data/labels" #"/scratch-grete/projects/nim00007/cryo-et/challenge-data/public_test_dataset/ground_truth_scaled_for_detection/" #"/scratch-grete/projects/nim00007/cryo-et/challenge-data/train/overlay/"
OUTPUT_ROOT = "/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/training" #"/mnt/vast-nhr/home/muth9/u12095/cryo-et/czii_challenge/training" #"/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/training"

def find_zarr_or_mrc(base_path):
    zarr_folders = []
    
    # Walk through all subdirectories recursively
    for root, dirs, files in os.walk(base_path):
        for d in dirs:
            if d.endswith(".zarr"):
                zarr_folders.append(os.path.join(root, d))
    
    if zarr_folders:
        # Prefer denoised.zarr if it exists
        for folder in zarr_folders:
            if os.path.basename(folder) == "denoised.zarr":
                return folder
        # Otherwise, return the first .zarr found
        return zarr_folders[0]
    
    # If no .zarr found, look for .mrc files
    mrc_files = []
    for root, dirs, files in os.walk(base_path):
        for f in files:
            if f.endswith(".mrc"):
                mrc_files.append(os.path.join(root, f))
    
    if not mrc_files:
        raise FileNotFoundError(f"No .zarr folder or .mrc file found in {base_path}")
    
    # Return the first .mrc found
    return mrc_files[0]

    
def train(key, ignore_label=None, training_2D=False, testset=True, extension="zarr"):

    datasets = ["ExperimentRuns", "ExperimentRuns_faket"]
    model_name = "protein_detection_czii_v15"

    output_path = os.path.join(OUTPUT_ROOT, model_name)
    os.makedirs(output_path, exist_ok=True)

    train_paths, train_label_paths = get_paths(
        "train", datasets=datasets, train_root=TRAIN_ROOT,
        output_root=output_path, testset=testset, label_root=LABEL_ROOT
    )
    val_paths, val_label_paths = get_paths(
        "val", datasets=datasets, train_root=TRAIN_ROOT,
        output_root=output_path, testset=testset, label_root=LABEL_ROOT
    )

    if testset:
        test_paths, test_label_paths = get_paths(
            "test", datasets=datasets, train_root=TRAIN_ROOT,
            output_root=output_path, testset=testset, label_root=LABEL_ROOT
        )
    else:
        test_paths, test_label_paths = None, None

    print("Start training with:")
    print(len(train_paths), "tomograms for training")
    print(len(val_paths), "tomograms for validation")

    patch_shape = [48, 256, 256]

    batch_size = 2
    check = False

    #add the zarr file path ending to each path
    train_paths = [find_zarr_or_mrc(path) for path in train_paths]
    val_paths = [find_zarr_or_mrc(path) for path in val_paths]
    test_paths = [find_zarr_or_mrc(path) for path in test_paths]

    print(f"train_paths {train_paths}")
    print(f"val_paths{val_paths}")
    print(f"test_paths {test_paths}")
    
    # TODO do we want n_samples_train and n_samples_val in the supervised training?
    supervised_training(
        name=model_name,
        train_paths=train_paths,
        train_label_paths=train_label_paths,
        val_paths=val_paths,
        val_label_paths=val_label_paths,
        raw_key = "0",
        patch_shape=patch_shape, batch_size=batch_size,
        check=check,
        lr=1e-4,
        n_iterations=1e4,
        out_channels=5,
        augmentations=None,
        eps=1e-5,
        sigma=None,
        lower_bound=None,
        upper_bound=None,
        test_paths=test_paths,
        test_label_paths=test_label_paths,
        save_root="/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/models", #"/mnt/vast-nhr/home/muth9/u12095/cryo-et/czii_challenge/models", #"/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/models",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--testset", action='store_false', help="Set to False if no testset should be created")
    args = parser.parse_args()
    train(args.testset)


if __name__ == "__main__":
    main()
