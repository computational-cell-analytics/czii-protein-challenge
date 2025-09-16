import os

import json
import numpy as np
import zarr
from glob import glob
from typing import List, Tuple, Sequence
from sklearn.model_selection import train_test_split
from numpy.typing import ArrayLike
from elf.io import open_file

from classification.data_processing import extract_subtomograms, get_max_extent


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

def get_non_zarr(input_path):
    #TODO expand for other file types

    import mrcfile

    # Look for .mrc files in the directory
    mrc_files = [f for f in os.listdir(input_path) if f.lower().endswith('.mrc')]
    
    if not mrc_files:
        raise FileNotFoundError(f"No .mrc file found in {input_path}")
    if len(mrc_files) > 1:
        raise ValueError(f"Multiple .mrc files found in {input_path}: {mrc_files}")
    
    # Get the single .mrc file
    mrc_path = os.path.join(input_path, mrc_files[0])

    # Open MRC file
    with mrcfile.open(mrc_path, permissive=True) as mrc:
        input_volume = mrc.data  

    return input_volume

def get_volume(input_path: str, zarr_: bool) -> np.ndarray:
    if zarr_:
        # Walk through directory tree and find the first .zarr folder
        zarr_dir = None
        for root, dirs, files in os.walk(input_path):
            for d in dirs:
                if d.endswith(".zarr"):
                    zarr_dir = os.path.join(root, d)
                    break
            if zarr_dir:
                break

        if zarr_dir is None:
            raise FileNotFoundError(f"No .zarr folder found under {input_path}")

        # Append "0" subfolder
        zarr_path = os.path.join(zarr_dir, "0")

        if not os.path.exists(zarr_path):
            raise FileNotFoundError(f"Expected '0' subfolder inside {zarr_dir}, but not found.")

        # Open and load volume
        zarr_file = zarr.open(zarr_path, mode="r")
        volume = zarr_file[:]
        
    else:
        volume = get_non_zarr(input_path)

    return volume


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

def parse_json_files(json_files):
    """
    Parse multiple JSON files to extract coordinates and protein names.

    Parameters:
        json_files (list of str): List of JSON file paths.

    Returns:
        list of tuples, list: Extracted coordinates and corresponding protein types.
    """
    coordinates = []
    protein_types = []

    for file in json_files:
        with open(file, 'r') as f:
            data = json.load(f)

            points = data.get("points", [])
            for point in points:
                location = point.get("location", {})
                x = location.get("x")
                y = location.get("y")
                z = location.get("z")
                if x is not None and y is not None and z is not None:
                    coordinates.append((z / 10, y / 10, x / 10))  # Scale coordinates as needed
                    protein_types.append(data.get("pickable_object_name", "unknown"))

    return coordinates, protein_types


def get_coords_and_targets(paths: List[str], target_root: str) -> Tuple[List[List[Tuple[int, int, int]]], List[List[str]]]:
    """
    For each tomogram path, load coordinates and targets from the matching experiment folder in target_root.

    Returns:
        coords_all: list where each element is a list of integer coordinates for that tomogram
        targets_all: list where each element is a list of protein type labels for that tomogram
    """
    coords_all = []
    targets_all = []

    for path in paths:
        experiment_name = os.path.basename(path)
        json_folder = os.path.join(target_root, experiment_name)
        picks_folder = os.path.join(json_folder, "Picks")

        # Load and parse JSONs, skipping albumin.json
        json_files = [
            os.path.join(picks_folder, f)
            for f in os.listdir(picks_folder)
            if f.endswith('.json') and f != "albumin.json" #need this for synthetic data
        ]
        
        coords, protein_types = parse_json_files(json_files)

        # Convert to integers for array slicing
        coords_int = [(int(round(x)), int(round(y)), int(round(z))) for x, y, z in coords]

        coords_all.append(coords_int)
        targets_all.append(protein_types)  # keep per-tomogram grouping

    return coords_all, targets_all

def get_data(
    paths: List[str],
    coords_all: List[List[Tuple[int, int, int]]],
    max_extent: int,
    in_channels: int,
    targets: List[List[str]],
    zarr_: bool,
):
    """
    Given a list of paths to tomograms, extract subtomograms using detection coordinates.

    Each subtomogram will have shape (in_channels, D, H, W).
    """
    subtomograms = []
    all_filtered_targets = []

    for path, coords, tomogram_targets in zip(paths, coords_all, targets):
        raw_volume = get_volume(path, zarr_)  # expected shape: (D, H, W)

        subs, _, filtered_targets = extract_subtomograms(
            raw_volume,
            coords,
            max_extent,
            targets=tomogram_targets
        )

        # Add channel dimension to each subtomogram
        for sub, tgt in zip(subs, filtered_targets):
            sub = np.expand_dims(sub, axis=0)  # shape: (1, D, H, W)
            if in_channels == 1:
                subtomograms.append(sub)
                all_filtered_targets.append(tgt)
            else:
                # Repeat the single channel across in_channels
                sub = np.repeat(sub, in_channels, axis=0)  # shape: (in_channels, D, H, W)
                subtomograms.append(sub)
                all_filtered_targets.append(tgt)

    return subtomograms, all_filtered_targets

def get_single_subtomogram(
    path: str,
    coord: Tuple[int, int, int],
    max_extent: int,
    in_channels: int,
    target: str,
    zarr_: bool,
):
    """
    Load a single subtomogram on demand from a tomogram file.
    """

    # ⚠️ If get_volume loads the *whole* tomogram, 
    # you may still use a lot of memory.
    # If the data is in Zarr, you can slice directly:
    raw_volume = get_volume(path, zarr_)  # (D, H, W)

    subs, _, filtered_targets = extract_subtomograms(
        raw_volume,
        [coord],       # single coordinate
        max_extent,
        targets=[target],
    )

    sub = subs[0]
    tgt = filtered_targets[0]

    # add channel dimension
    sub = np.expand_dims(sub, axis=0)  # (1, D, H, W)
    if in_channels > 1:
        sub = np.repeat(sub, in_channels, axis=0)

    return sub, tgt



