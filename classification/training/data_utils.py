import os
import sys
sys.path.append("/user/muth9/u12095/czii-protein-challenge")
import json
import numpy as np
import zarr
from glob import glob
from typing import List, Tuple, Sequence
from sklearn.model_selection import train_test_split
from numpy.typing import ArrayLike

from classification.data_processing.create_subtomograms import extract_subtomograms, get_max_extent


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

def compute_max_extent_from_all(
    all_peaks: List[Sequence[Tuple[int, int, int]]],
    all_heatmaps: List[np.ndarray],
) -> Tuple[int, int, int]:
    max_extents = [
        get_max_extent(peaks, heatmap)
        for peaks, heatmap in zip(all_peaks, all_heatmaps)
    ]
    max_extent = max(max_extents)
    return max_extent

def get_coords_and_heatmaps(
    paths: List[str],
    detection_root: str
) -> Tuple[List[List[Tuple[int, int, int]]], List[np.ndarray]]:
    coords_all = []
    heatmaps = []

    for path in paths:
        experiment_name = os.path.basename(path)

        peaks_path = os.path.join(detection_root, f"{experiment_name}_protein_detections.json")
        coords = load_peaks(peaks_path)
        coords_all.append(coords)

        heatmap_path = os.path.join(detection_root, f"{experiment_name}_protein_detections.npy")
        heatmap = load_heatmap(heatmap_path)
        heatmaps.append(heatmap)

    return coords_all, heatmaps

def get_data(
    paths: List[str],
    coords_all: List[List[Tuple[int, int, int]]],
    max_extent: int,
    in_channels: int
) -> Sequence[ArrayLike]:
    """
    Given a list of paths to tomograms, extract subtomograms using detection coordinates.

    Each subtomogram will have shape (in_channels, D, H, W).
    """
    subtomograms = []

    for path, coords in zip(paths, coords_all):
        raw_volume = get_volume(path)  # expected shape: (D, H, W)
        subs, _ = extract_subtomograms(raw_volume, coords, max_extent)  # list of (D, H, W)

        # Add channel dimension to each subtomogram
        for sub in subs:
            sub = np.expand_dims(sub, axis=0)  # shape: (1, D, H, W)
            if in_channels == 1:
                subtomograms.append(sub)
            else:
                # Repeat the single channel across in_channels
                sub = np.repeat(sub, in_channels, axis=0)  # shape: (in_channels, D, H, W)
                subtomograms.append(sub)

    return subtomograms



