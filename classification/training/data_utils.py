import os
import json
import numpy as np
import zarr
from glob import glob
from typing import List, Tuple
from sklearn.model_selection import train_test_split


def _train_val_test_split(names: List[str], train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    train, test_val = train_test_split(names, test_size=1 - train_ratio, shuffle=True)
    val_size = test_ratio / (val_ratio + test_ratio)
    val, test = train_test_split(test_val, test_size=val_size)
    return train, val, test


def _train_val_split(names: List[str], train_ratio=0.8):
    return train_test_split(names, test_size=1 - train_ratio, shuffle=True)


def _get_subdirs(path: str) -> List[str]:
    return sorted([name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))])


def _require_train_val_test_split(datasets: List[str], train_root: str, output_root: str) -> None:
    for ds in datasets:
        split_path = os.path.join(output_root, f"split-{ds}.json")
        if os.path.exists(split_path):
            continue

        ds_path = os.path.join(train_root, ds)
        file_names = _get_subdirs(ds_path)

        train, val, test = _train_val_test_split(file_names)

        with open(split_path, "w") as f:
            json.dump({"train": train, "val": val, "test": test}, f)


def _require_train_val_split(datasets: List[str], train_root: str, output_root: str) -> None:
    for ds in datasets:
        split_path = os.path.join(output_root, f"split-{ds}.json")
        if os.path.exists(split_path):
            continue

        ds_path = os.path.join(train_root, ds)
        file_names = _get_subdirs(ds_path)

        train, val = _train_val_split(file_names)

        with open(split_path, "w") as f:
            json.dump({"train": train, "val": val}, f)


def get_paths(
    split: str,
    datasets: List[str],
    train_root: str,
    output_root: str,
    testset: bool = True
) -> List[str]:
    if testset:
        _require_train_val_test_split(datasets, train_root, output_root)
    else:
        _require_train_val_split(datasets, train_root, output_root)

    paths = []
    for ds in datasets:
        split_path = os.path.join(output_root, f"split-{ds}.json")
        with open(split_path) as f:
            names = json.load(f)[split]

        ds_paths = [os.path.join(train_root, ds, name) for name in names]

        assert all(os.path.exists(p) for p in ds_paths), f"Missing paths in: {ds_paths}"
        paths.extend(ds_paths)

    return paths


def get_volume(input_path: str) -> np.ndarray:
    zarr_path = os.path.join(input_path, "VoxelSpacing10.000", "denoised.zarr", "0")
    zarr_file = zarr.open(zarr_path, mode="r")
    return zarr_file[:]


def load_heatmap(npy_filepath: str) -> np.ndarray:
    heatmap = np.load(npy_filepath)
    return np.squeeze(heatmap)


def load_peaks(json_filepath: str) -> List[Tuple[int, int, int]]:
    with open(json_filepath, "r") as f:
        peaks = json.load(f)
    return [tuple(map(int, peak)) for peak in peaks]


def get_data(
    split: str,
    datasets: List[str],
    train_root: str,
    detection_root: str,
    output_root: str,
    n_classes: int,
    seed: int = 42,
    testset: bool = True
) -> Tuple[
    List[np.ndarray],
    List[List[Tuple[int, int, int]]],
    List[np.ndarray]
]:
    np.random.seed(seed)

    paths = get_paths(split, datasets, train_root, output_root, testset)

    raw_volumes = []
    coords_all = []
    heatmaps = []

    for path in paths:
        raw_volumes.append(get_volume(path))

        experiment_name = os.path.basename(path)
        peaks_path = f"{detection_root}/{experiment_name}_protein_detections.json"
        coords_all.append(load_peaks(peaks_path))

        heatmap_path = f"{detection_root}/{experiment_name}_protein_detections.npy"
        heatmaps.append(load_heatmap(heatmap_path))


    return raw_volumes, coords_all, heatmaps