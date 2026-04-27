import zarr
import numpy as np
import os
import h5py
import json

from typing import List, Tuple, Optional
from collections import defaultdict, deque
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from classification.data_processing import extract_subtomograms
from classification.training import get_coords_and_targets


def load_detection_predictions(pred_path):
    with open(pred_path, "r") as f:
        points = json.load(f)
    return np.array(points)


def get_non_zarr(input_path):
    """
    Load a single volumetric file from a directory.
    Supports .mrc, .h5, .npy, .tif/.tiff.
    """

    files = os.listdir(input_path)
    
    #Supported extensions
    supported_exts = ['.mrc', '.h5', '.npy', '.tif', '.tiff']
    
    # Find files with supported extensions
    valid_files = [f for f in files if os.path.splitext(f)[1].lower() in supported_exts]
    
    if not valid_files:
        raise FileNotFoundError(f"No supported files found in {input_path}. Supported extensions: {supported_exts}")

    file_path = os.path.join(input_path, valid_files[0])
    ext = os.path.splitext(file_path)[1].lower()
    
    # Load depending on file type
    if ext == '.mrc':
        import mrcfile
        with mrcfile.open(file_path, permissive=True) as mrc:
            volume = mrc.data
    elif ext in ['.tif', '.tiff']:
        from tifffile import imread
        volume = imread(file_path)
    elif ext == '.npy':
        volume = np.load(file_path)
    elif ext == '.h5':
        from elf.io import open_file
        with open_file(input_path, "r") as f:

            # Try to automatically derive the key with the raw data.
            keys = list(f.keys())
            if len(keys) == 1:
                key = keys[0]
            elif "data" in keys:
                key = "data"
            elif "raw" in keys:
                key = "raw"

            volume = f[key][:]
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    
    return volume


def get_volume(input_path: str, zarr_: bool) -> np.ndarray:
    if zarr_:
        # Recursive search for .zarr folders
        zarr_folders = []
        for root, dirs, files in os.walk(input_path):
            for d in dirs:
                if d.endswith(".zarr"):
                    zarr_folders.append(os.path.join(root, d))
        
        if not zarr_folders:
            raise FileNotFoundError(f"No .zarr folder found under {input_path}")
        
        # Prefer denoised.zarr if it exists
        zarr_dir = next((f for f in zarr_folders if os.path.basename(f) == "denoised.zarr"), zarr_folders[0])
        
        # Append "0" subfolder
        zarr_path = os.path.join(zarr_dir, "0")
        print(f"Using volume path: {zarr_path}")
        if not os.path.exists(zarr_path):
            raise FileNotFoundError(f"Expected '0' subfolder inside {zarr_dir}, but not found.")
        
        # Open and load volume
        zarr_file = zarr.open(zarr_path, mode="r")
        volume = zarr_file[:]
        
    else:
        volume = get_non_zarr(input_path)
    
    return volume


def match_predictions_to_labels(preds, gts, match_distance=45):
    """
    Match predicted coordinates to ground-truth coordinates using Hungarian algorithm.
    
    Args:
        preds: np.ndarray of shape (M, D) - predicted coordinates
        gts: np.ndarray of shape (N, D) - ground-truth coordinates
        match_distance: float, maximum distance to consider a match.
    
    Returns:
        matched_pred_idx: list of indices in preds that were matched
        matched_gt_idx: list of indices in gts that were matched
    """
    n, m = len(gts), len(preds)

    if n == 0 or m == 0:
        return [], []

    # Compute pairwise distances
    pairwise_distances = cdist(gts, preds, metric="euclidean")

    # Define costs (Hungarian finds 'minimum' cost)
    max_distance = pairwise_distances.max() if pairwise_distances.size > 0 else 1.0
    costs = -(pairwise_distances < match_distance).astype(float) - \
            (max_distance - pairwise_distances) / max_distance

    # Hungarian matching
    gt_idx, pred_idx = linear_sum_assignment(costs)

    # Keep only valid matches under distance threshold
    match_mask = pairwise_distances[gt_idx, pred_idx] < match_distance
    matched_gt_idx = gt_idx[match_mask].tolist()
    matched_pred_idx = pred_idx[match_mask].tolist()

    return matched_pred_idx, matched_gt_idx


def match_preds_with_labels(preds, label_path, no_class_label="no_class"):
    """
    Match detection predictions with GT labels, but:
    - GT points labeled 'no_class' are NOT included in matching
    - Unmatched predictions get the label 'no_class'

    Returns:
        matched_coords: list of coordinates (one per prediction, including unmatched)
        matched_labels: list of GT labels (same length as matched_coords)
        num_unmatched_true: number of predictions that got no_class assigned
    """
    # Load and parse JSONs, skipping albumin.json, actin.json, and mt.json
    json_files = [
        os.path.join(label_path, f)
        for f in os.listdir(label_path)
        if f.endswith('.json') and f != "albumin.json" and f != "actin.json" and f != "mt.json"#need this for synthetic data
    ]
    from detection.data_processing.create_heatmap import parse_json_files
    label_coords, protein_types = parse_json_files(json_files)

    label_coords = np.array(label_coords)
    protein_types = np.array(protein_types)

    #Remove GT points labeled no_class (they cannot match)
    keep_mask = protein_types != no_class_label
    real_gt_coords = label_coords[keep_mask]
    real_gt_labels = protein_types[keep_mask]

    #Run matching 
    matched_idx_pred, matched_idx_gt = match_predictions_to_labels(preds, real_gt_coords)

    filter = False
    match_distance = 45

    if filter:
        # Compute nearest GT distance for each pred
        if len(real_gt_coords) > 0:
            dists = cdist(preds, real_gt_coords).min(axis=1)
        else:
            dists = np.full(len(preds), np.inf)

        dists = np.array(dists)

        is_matched = np.zeros(len(preds), dtype=bool)
        is_matched[matched_idx_pred] = True

        #filter matched preds & preds with no GT within 45
        keep_mask = is_matched | (dists >= match_distance)

        preds = np.array(preds)[keep_mask]
        dists = dists[keep_mask]

        #labels: matched GT label or no_class
        assigned_labels = np.array([no_class_label] * len(preds), dtype=object)

        old_to_new = {old: new for new, old in enumerate(np.where(keep_mask)[0])}

        for old_p, old_gt in zip(matched_idx_pred, matched_idx_gt):
            if old_p in old_to_new:
                new_p = old_to_new[old_p]
                assigned_labels[new_p] = real_gt_labels[old_gt]

        num_unmatched = np.sum(assigned_labels == no_class_label)

        #integer coords
        coords = np.round(preds).astype(int)

        return coords, assigned_labels.tolist(), num_unmatched, dists.tolist()
    
    else:
        #create arrays for final output
        num_preds = len(preds)
        assigned_labels = [no_class_label] * num_preds   # default label for all predictions
        matched_coords = [preds[i] for i in range(num_preds)]  # keep all preds
        matched_coords = np.round(matched_coords).astype(int)

        # assign matched labels
        for p_idx, gt_idx in zip(matched_idx_pred, matched_idx_gt):
            assigned_labels[p_idx] = real_gt_labels[gt_idx]

        # distances from every pred to nearest real GT point
        if len(real_gt_coords) > 0:
            dists = cdist(preds, real_gt_coords).min(axis=1)
        else:
            dists = np.full(num_preds, np.inf)

        #Count predictions that were not matched
        num_unmatched_true = sum(1 for lbl in assigned_labels if lbl == no_class_label)

        return matched_coords, assigned_labels, num_unmatched_true, dists


def preprocess_tomo_with_predictions(
    tomo_paths: List[str],
    pred_path: str,
    label_path: str,
    max_extent: int,
    subtomo_output: str,
    halo: Optional[int] = None
) -> Tuple[List[str], List[str], int, dict]:
    """
    Extract subtomograms at labeled coordinates and save them as .h5 files.

    Returns:
        file_paths: list of .h5 file paths
        labels: list of corresponding ground-truth labels (one per file)
        num_unmatched: total count of no_class assignments (global)
        unmatched_info: dict per tomogram
    """
    # Load predictions and match to GT (global lists)
    preds_global = load_detection_predictions(pred_path)          # shape (P,3)
    coords_global, targets_global, num_unmatched, nearest_gt_distances = match_preds_with_labels(preds_global, label_path)

    # Ensure arrays
    preds_global = np.array(preds_global)
    coords_global = np.round(np.array(coords_global)).astype(int)  # integer coords aligned to preds
    nearest_gt_distances = np.array(nearest_gt_distances)

    # Build mapping from coord tuple -> queue of indices in the global predictions array.
    coord_to_indices = defaultdict(deque)
    for idx, c in enumerate(coords_global):
        coord_to_indices[(int(c[0]), int(c[1]), int(c[2]))].append(idx)

    os.makedirs(subtomo_output, exist_ok=True)

    file_paths = []
    labels = []

    # Store unmatched pred data per tomogram
    unmatched_info = {}

    for tomo_path in tomo_paths:
        experiment_name = os.path.basename(tomo_path)
        raw_volume = get_volume(tomo_path, zarr_=True)

        # Convert from (x, y, z) to (z, y, x)
        coords_for_extract = [(int(c[0]), int(c[1]), int(c[2])) for c in coords_global]


        # extract subtomograms: it will return only those coords that are valid in this tomo
        if halo is not None:
            subtomograms, valid_coords, targets = extract_subtomograms(raw_volume, coords_for_extract, max_extent, halo=halo, targets=targets_global)
        else:
            subtomograms, valid_coords, targets = extract_subtomograms(raw_volume, coords_for_extract, max_extent, targets=targets_global)


        # Map valid_coords back to their indices in the global predictions (using the deque)
        indices_in_global = []
        for vc in valid_coords:
            if vc in coord_to_indices and coord_to_indices[vc]:
                indices_in_global.append(coord_to_indices[vc].popleft())
            else:
                # If there's no mapping, append None (shouldn't usually happen)
                indices_in_global.append(None)

        #per-valid distances aligned with valid_coords order
        per_valid_distances = []
        for idx in indices_in_global:
            if idx is None:
                per_valid_distances.append(np.inf)
            else:
                per_valid_distances.append(float(nearest_gt_distances[idx]))
        per_valid_distances = np.array(per_valid_distances)

        #unmatched point info for this tomogram
        unmatched_mask = np.array(targets) == "no_class"
        unmatched_coords = np.array(valid_coords)[unmatched_mask]
        unmatched_distances = per_valid_distances[unmatched_mask]

        unmatched_info[experiment_name] = {
            "coords": unmatched_coords.tolist(),
            "distances": unmatched_distances.tolist(),
        }

        # Save subtomograms and labels (valid_coords are (z,y,x) returned by extract_subtomograms)
        for cube, (z, y, x), target in zip(subtomograms, valid_coords, targets):
            filename = f"{experiment_name}_x{x}_y{y}_z{z}.h5"
            filepath = os.path.join(subtomo_output, filename)

            with h5py.File(filepath, "w") as f:
                f.create_dataset("raw", data=cube, compression="lzf")

            file_paths.append(filepath)
            labels.append(target)

    return file_paths, labels, num_unmatched, unmatched_info


def preprocess_tomo_with_labels(
    tomo_paths: List[str],
    labels_root: str,
    max_extent: int,
    subtomo_output: str,
    halo: Optional[int] = None
) -> Tuple[List[str], List[str]]:
    """
    Extract subtomograms at labeled coordinates and save them as .h5 files.

    Returns:
        file_paths: list of .h5 file paths
        labels: list of corresponding ground-truth labels (one per file)
    """
    coords_all, targets_all = get_coords_and_targets(tomo_paths, labels_root)
    os.makedirs(subtomo_output, exist_ok=True)

    file_paths = []
    labels = []

    for tomo_path, coords, targets in zip(tomo_paths, coords_all, targets_all):
        experiment_name = os.path.basename(tomo_path)
        raw_volume = get_volume(tomo_path, zarr_=True)

        if halo is not None:
            subtomograms, valid_coords, targets = extract_subtomograms(raw_volume, coords, max_extent, halo=halo, targets=targets)
        else:
            subtomograms, valid_coords, targets = extract_subtomograms(raw_volume, coords, max_extent, targets=targets)

        for cube, (x, y, z), target in zip(subtomograms, valid_coords, targets):
            filename = f"{experiment_name}_x{x}_y{y}_z{z}.h5"
            filepath = os.path.join(subtomo_output, filename)

            with h5py.File(filepath, "w") as f:
                f.create_dataset("raw", data=cube, compression="lzf")

            file_paths.append(filepath)
            labels.append(target)

    return file_paths, labels
