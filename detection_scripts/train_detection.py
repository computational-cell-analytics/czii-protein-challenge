import os
import argparse
import json
from torch_em.segmentation import DEFAULT_SCHEDULER_KWARGS

from detection.utils import get_paths, reuse_split  
from detection.utils import supervised_training 
from detection.data_processing import save_max_extent_to_json

TRAIN_ROOT = "/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/data/"
LABEL_ROOT = "/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/ground_truth/structure_for_detection/"
OUTPUT_ROOT = "/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/training"


# Supported single-file tomogram formats (everything except .zarr, which is a folder).
NON_ZARR_EXTS = (".mrc", ".h5", ".hdf5", ".tif", ".tiff")

# Two orders of magnitude of annealing, enforced as a floor. Without it ReduceLROnPlateau keeps
# halving indefinitely: v39 reached 1.2e-8 and stopped learning with 40% of its budget unspent.
LR_DECADES = 2


def scheduler_kwargs_with_floor(lr):
    return {**DEFAULT_SCHEDULER_KWARGS, "min_lr": lr * 10 ** -LR_DECADES}


def find_tomogram(base_path):
    """Resolve a tomogram directory to its actual data path.

    Prefers a ``.zarr`` folder (``denoised.zarr`` if present), otherwise returns
    the first supported single file (.mrc/.h5/.hdf5/.tif/.tiff).
    """
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

    # If no .zarr found, look for a supported single file (.mrc, .h5, .tif, ...)
    data_files = []
    for root, dirs, files in os.walk(base_path):
        for f in files:
            if os.path.splitext(f)[1].lower() in NON_ZARR_EXTS:
                data_files.append(os.path.join(root, f))

    if not data_files:
        raise FileNotFoundError(
            f"No .zarr folder or supported file ({NON_ZARR_EXTS}) found in {base_path}"
        )

    # Return the first supported file found
    return data_files[0]


# Backwards-compatible alias.
find_zarr_or_mrc = find_tomogram


def train(key=None, ignore_label=None, training_2D=False, testset=True, extension="zarr",
          save_max_extent=True, model_name=None, patch_shape=None, split_from=None):

    datasets = ["ExperimentRuns_faket_dens1_5_distr_eqCl4"]
    synthetic_dataset = ["ExperimentRuns_faket_dens1_5_distr_eqCl4"] #used to save the max_extent information for the classification later
    model_name = model_name or "protein_detection_czii_v44"
    # z >= the U-Net's ~105 voxel z receptive field, and equal to the network input z that
    # get_default_tiling() picks for an >=80 GB GPU, so training and prediction see the same
    # amount of context. Revisit if inference runs on a smaller GPU (tile z drops to 96 / 80).
    patch_shape = patch_shape or [112, 256, 256]

    print(f"Training model {model_name} with patch shape {patch_shape}")

    output_path = os.path.join(OUTPUT_ROOT, model_name)
    os.makedirs(output_path, exist_ok=True)
    if split_from:
        reuse_split(split_from, output_path)

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

    batch_size = 2
    n_iterations = 5e3
    lr = 1e-4
    check = False

    # Resolve each tomogram directory to its actual data path (zarr / mrc / h5 / tif).
    train_paths = [find_tomogram(path) for path in train_paths]
    val_paths = [find_tomogram(path) for path in val_paths]
    test_paths = [find_tomogram(path) for path in test_paths] if test_paths is not None else None

    # The raw key only applies to container formats. Zarr stores the volume at
    # multiscale level "0"; for other formats let the loader pick the dataset.
    raw_key = "0" if all(p.endswith(".zarr") for p in train_paths) else None

    print(f"train_paths {train_paths}")
    print(f"val_paths{val_paths}")
    print(f"test_paths {test_paths}")

    # Epoch length is left to DetectionDataset.compute_len(), i.e. one tiling of each tomogram
    # (torch_em's default). Note this makes the epoch count -- and therefore the LR decay rate --
    # a function of patch_shape, so runs with different patches are not schedule-comparable.
    scheduler_kwargs = scheduler_kwargs_with_floor(lr)

    if save_max_extent:
        save_max_extent_to_json(TRAIN_ROOT, synthetic_dataset)
        
    supervised_training(
        name=model_name,
        train_paths=train_paths,
        train_label_paths=train_label_paths,
        val_paths=val_paths,
        val_label_paths=val_label_paths,
        raw_key=raw_key,
        patch_shape=patch_shape, batch_size=batch_size,
        scheduler_kwargs=scheduler_kwargs,
        check=check,
        lr=lr,
        n_iterations=n_iterations,
        out_channels=5,
        augmentations=None,
        eps=1e-5,
        sigma=None,
        lower_bound=None,
        upper_bound=None,
        test_paths=test_paths,
        test_label_paths=test_label_paths,
        save_root="/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/models",
        num_workers=8, #TODO maybe can go bigger here
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--testset", action='store_false', help="Set to False if no testset should be created. " \
                        "Only set to False if --no_max_extent is also set to False.")
    parser.add_argument("-me", "--save_max_extent", action='store_false', 
                        help="Set to False if no max extent (needed for classification) should not be saved automatically.")
    parser.add_argument("-n", "--name", default=None,
                        help="Checkpoint name. Defaults to the one set in train().")
    parser.add_argument("-p", "--patch_shape", nargs=3, type=int, default=None, metavar=("Z", "Y", "X"),
                        help="Training patch shape (z y x). Defaults to the one set in train().")
    parser.add_argument("--split_from", default=None,
                        help="Training output dir of a previous run whose split-*.json to reuse, so runs are "
                             "compared on identical train/val/test tomograms.")
    args = parser.parse_args()
    train(
        testset=args.testset,
        save_max_extent=args.save_max_extent,
        model_name=args.name,
        patch_shape=args.patch_shape,
        split_from=args.split_from,
    )


if __name__ == "__main__":
    main()
