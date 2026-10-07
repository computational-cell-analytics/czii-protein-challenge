import zarr
import numpy as np
import os
import h5py
import json

from typing import List, Tuple, Optional
from collections import defaultdict, deque
from scipy.spatial.distance import cdist

from classification.data_processing import extract_subtomograms
from classification.training import get_coords_and_targets
from detection.utils.evaluation.evaluation_metrics import hungarian_match


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
        with open_file(file_path, "r") as f:

            # Try to automatically derive the key with the raw data.
            keys = list(f.keys())
            if len(keys) == 1:
                key = keys[0]
            elif "data" in keys:
                key = "data"
            elif "raw" in keys:
                key = "raw"
            else:
                key = keys[0]

            volume = f[key][:]
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    
    return volume


def get_volume(input_path: str, zarr_: bool = None) -> np.ndarray:
    """Load a tomogram, auto-detecting the format.

    If a ``.zarr`` folder is present under ``input_path`` it is used, otherwise
    the loader falls back to a single ``.mrc/.h5/.npy/.tif/.tiff`` file
    (see :func:`get_non_zarr`). The ``zarr_`` argument is kept for backwards
    compatibility but is ignored: the format is detected automatically.
    """
    # Recursive search for .zarr folders
    zarr_folders = [
        os.path.join(root, d)
        for root, dirs, _ in os.walk(input_path)
        for d in dirs
        if d.endswith(".zarr")
    ]

    if zarr_folders:
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
        # Fallback to non-zarr loader
        volume = get_non_zarr(input_path)

    return volume


def match_predictions_to_labels(preds, gts, match_distance=None, gt_classes=None):
    """
    Match predicted coordinates to ground-truth coordinates using Hungarian algorithm.

    Args:
        preds: np.ndarray of shape (M, D) - predicted coordinates (voxels)
        gts: np.ndarray of shape (N, D) - ground-truth coordinates (voxels)
        match_distance: maximum distance to consider a match; by default the CZII criterion
            0.5 * particle radius, looked up per point from `gt_classes`.
        gt_classes: class label per ground-truth point.

    Returns:
        matched_pred_idx: list of indices in preds that were matched
        matched_gt_idx: list of indices in gts that were matched
    """
    gt_idx, pred_idx = hungarian_match(gts, preds, match_distance, gt_classes)
    return pred_idx.tolist(), gt_idx.tolist()


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

    # Run matching at the CZII tolerance (0.5 * particle radius, per class), so a crop only
    # inherits a label when the particle is actually near the centre of the crop.
    matched_idx_pred, matched_idx_gt = match_predictions_to_labels(
        preds, real_gt_coords, gt_classes=real_gt_labels
    )

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
) -> Tuple[List[str], List[str], dict, dict]:
    """
    Extract subtomograms at labeled coordinates and save them as .h5 files.

    Returns:
        file_paths: list of .h5 file paths
        labels: list of corresponding ground-truth labels (one per file)
        counts: prediction bookkeeping, counted *after* the border crops are dropped
        unmatched_info: dict per tomogram
    """
    # Load predictions and match to GT (global lists)
    preds_global = load_detection_predictions(pred_path)          # shape (P,3)
    coords_global, targets_global, _, nearest_gt_distances = match_preds_with_labels(preds_global, label_path)

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
        raw_volume = get_volume(tomo_path)

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

    # Count on the crops that actually reach the classifier: extract_subtomograms silently
    # drops predictions whose box would leave the volume, so the no_class count from the
    # matching step is not the number of no_class crops.
    unmatched_kept = sum(1 for lbl in labels if lbl == "no_class")
    counts = {
        "total_preds": len(preds_global),
        "kept_crops": len(file_paths),
        "matched_preds": len(file_paths) - unmatched_kept,
        "unmatched_preds": unmatched_kept,
        # exact for the single-tomogram case both callers use
        "dropped_at_border": len(preds_global) - len(file_paths),
    }

    return file_paths, labels, counts, unmatched_info


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
        raw_volume = get_volume(tomo_path)

        if halo is not None:
            subtomograms, valid_coords, targets = extract_subtomograms(raw_volume, coords, max_extent, halo=halo, targets=targets)
        else:
            subtomograms, valid_coords, targets = extract_subtomograms(raw_volume, coords, max_extent, targets=targets)

        # valid_coords are (z, y, x), as returned by extract_subtomograms
        for cube, (z, y, x), target in zip(subtomograms, valid_coords, targets):
            filename = f"{experiment_name}_x{x}_y{y}_z{z}.h5"
            filepath = os.path.join(subtomo_output, filename)

            with h5py.File(filepath, "w") as f:
                f.create_dataset("raw", data=cube, compression="lzf")

            file_paths.append(filepath)
            labels.append(target)

    return file_paths, labels
