import os
import shutil
from glob import glob
import json
from sklearn.model_selection import train_test_split


def reuse_split(source_root, output_root):
    """Copy split-*.json from a previous run so a new run gets the identical tomogram split.

    `_require_*_split` otherwise draws a fresh unseeded split per output dir, which would put
    a previous run's test tomograms into this run's training set.
    """
    if not os.path.isdir(source_root):
        raise FileNotFoundError(f"No split source directory: {source_root}")

    copied = []
    for name in sorted(os.listdir(source_root)):
        if name.startswith("split-") and name.endswith(".json"):
            target = os.path.join(output_root, name)
            if not os.path.exists(target):
                shutil.copy(os.path.join(source_root, name), target)
            copied.append(name)

    if not copied:
        raise FileNotFoundError(f"No split-*.json found in {source_root}")
    print(f"Reusing split from {source_root}: {copied}")
    return copied


def _require_train_val_test_split(datasets, train_root, output_root, extension):
    train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1

    def _train_val_test_split(names):
        train, test = train_test_split(names, test_size=1 - train_ratio, shuffle=True)
        _ratio = test_ratio / (test_ratio + val_ratio)
        val, test = train_test_split(test, test_size=_ratio)
        return train, val, test

    for ds in datasets:
        print(ds)
        split_path = os.path.join(output_root, f"split-{ds}.json")
        if os.path.exists(split_path):
            continue

        ds_path = os.path.join(train_root, ds)
        # need to check if the dataset contains files or folders (like eg for zarr) and ignore json files
        if any(
            os.path.isfile(os.path.join(ds_path, f)) and not f.endswith(".json")
            for f in os.listdir(ds_path)
        ):
            # If the dataset contains files
            file_paths = sorted(glob(os.path.join(ds_path, f"*.{extension}")))
            file_names = [os.path.basename(path) for path in file_paths]
        else:
            # If the dataset contains folders
            file_names = sorted(next(os.walk(ds_path))[1])  # Get subfolder names

        train, val, test = _train_val_test_split(file_names)

        with open(split_path, "w") as f:
            json.dump({"train": train, "val": val, "test": test}, f)


def _require_train_val_split(datasets, train_root, output_root, extension):
    train_ratio, val_ratio = 0.8, 0.2

    def _train_val_split(names):
        train, val = train_test_split(names, test_size=1 - train_ratio, shuffle=True)
        return train, val

    for ds in datasets:
        print(ds)
        split_path = os.path.join(output_root, f"split-{ds}.json")
        if os.path.exists(split_path):
            continue

        ds_path = os.path.join(train_root, ds)
        # need to check if the dataset contains files or folders (like eg for zarr)
        if any(os.path.isfile(os.path.join(ds_path, f)) for f in os.listdir(ds_path)):
            # If the dataset contains files
            file_paths = sorted(glob(os.path.join(ds_path, f"*.{extension}")))
            file_names = [os.path.basename(path) for path in file_paths]
        else:
            # If the dataset contains folders
            file_names = sorted(next(os.walk(ds_path))[1])  # Get subfolder names

        train, val = _train_val_split(file_names)

        with open(split_path, "w") as f:
            json.dump({"train": train, "val": val}, f)


def get_paths(split, datasets, train_root, output_root, testset=True, extension="zarr", label_root=None):
    if testset:
        _require_train_val_test_split(datasets, train_root, output_root, extension)
    else:
        _require_train_val_split(datasets, train_root, output_root, extension)

    paths = []
    label_paths = []
    for ds in datasets:
        split_path = os.path.join(output_root, f"split-{ds}.json")
        with open(split_path) as f:
            names = json.load(f)[split]

        ds_path = os.path.join(train_root, ds)
        if any(os.path.isfile(os.path.join(ds_path, f)) for f in os.listdir(ds_path)):
            # Paths for files
            ds_paths = [os.path.join(ds_path, name) for name in names]
        else:
            # Paths for folders
            ds_paths = [os.path.join(ds_path, name) for name in names]

        # Label paths for the current dataset
        if label_root:
            ds_label_path = os.path.join(label_root, ds)
            if any(os.path.isfile(os.path.join(ds_label_path, f)) for f in os.listdir(ds_label_path)):
                # File-based labels
                ds_label_paths = [os.path.join(ds_label_path, name) for name in names]
            else:
                # Folder-based labels
                ds_label_paths = [os.path.join(ds_label_path, name) for name in names]
        else:
            ds_label_paths = []

        assert all(os.path.exists(path) for path in ds_paths)
        if label_root:
            assert all(os.path.exists(path) for path in ds_label_paths)

        paths.extend(ds_paths)
        label_paths.extend(ds_label_paths)

    return paths, label_paths
