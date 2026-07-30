import pandas
import time
from tqdm import tqdm
import zarr
import json
import os
from skimage.feature import blob_log, peak_local_max
from ..evaluation.evaluation_metrics import metric_coords
from ..prediction.prediction import get_prediction_torch_em, load_detection_model
from ..training.tiling_helper import parse_tiling
from detection.data_processing.create_heatmap import parse_json_files
import numpy as np

from detection.config import (
    ADJ_FACTOR,
    CZII_SMALLEST_PROTEIN_SIZE,
    GRIDSEARCH_BETA,
    GRIDSEARCH_THRESH_RANGE,
)

#TODO Do I want to make this more flexible??
TRAIN_ROOT = "/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/data/"
LABEL_ROOT = "/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/ground_truth/structure_for_detection/"
DEFAULT_JSON = "/mnt/lustre-emmy-hdd/usr/u12095/cryo-et/czii_challenge/training/protein_detection_czii_v4/split-ExperimentRuns.json"


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


def get_volume(input_path: str) -> np.ndarray:
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
        if not os.path.exists(zarr_path):
            raise FileNotFoundError(f"Expected '0' subfolder inside {zarr_dir}, but not found.")
        
        # Open and load volume
        zarr_file = zarr.open(zarr_path, mode="r")
        volume = zarr_file[:]
    else:
        # Fallback to non-zarr loader
        volume = get_non_zarr(input_path)
    
    return volume


def get_full_image_path(json_val_path, val_path):
    file_name = os.path.basename(json_val_path)
    # Remove the prefix "split-" and the suffix ".json"
    experiment_name = file_name[len("split-"):-len(".json")]

    image_path = os.path.join(TRAIN_ROOT, experiment_name,val_path)

    return image_path


def get_full_label_path(json_val_path, val_path):
    file_name = os.path.basename(json_val_path)
    # Remove the prefix "split-" and the suffix ".json"
    experiment_name = file_name[len("split-"):-len(".json")]

    base_label_path = os.path.join(LABEL_ROOT, experiment_name, val_path)

    # Check if "Picks" folder exists, use that path if it does
    picks_path = os.path.join(base_label_path, "Picks")
    if os.path.exists(picks_path) and os.path.isdir(picks_path):
        return picks_path
    else:
        return base_label_path


def gridsearch(json_val_path, model_path):
    print("starting grid search")

    threshes = np.arange(*GRIDSEARCH_THRESH_RANGE)
    min_thresh = float(threshes.min())
    min_distance = int(CZII_SMALLEST_PROTEIN_SIZE * ADJ_FACTOR)
    data = []

    # Load JSON from the file
    with open(json_val_path, "r") as file:
        json_data = json.load(file)

    # Extract the 'val' list
    val_list = json_data["val"]

    # Load the model once and reuse it for every validation tomogram.
    model = load_detection_model(model_path)

    for val_path in val_list:

        image_path = get_full_image_path(json_val_path= json_val_path, val_path=val_path)
        label_path = get_full_label_path(json_val_path= json_val_path, val_path=val_path)
        print(f"double check image path {image_path} and label path {label_path}")

        tiling = parse_tiling(tile_shape=None, halo=None) #TODO implement tiling and halo choices

        input_volume = get_volume(image_path)
        pred = get_prediction_torch_em(input_volume=input_volume, tiling=tiling, model=model, verbose=True)[0]

        json_files = [
            os.path.join(label_path, f)
            for f in os.listdir(label_path)
            if f.endswith('.json') and f not in ('no_class.json', 'albumin.json', "actin.json", "mt.json")
        ]
        label_coords, _ = parse_json_files(json_files)

        # peak_local_max is expensive, so run it once at the lowest threshold. Raising
        # threshold_abs only removes peaks below it (it never revives peaks that were
        # suppressed by a stronger neighbour), so the peaks for any higher threshold are
        # exactly the subset of these peaks whose intensity clears that threshold.
        all_peaks = peak_local_max(pred, min_distance=min_distance, threshold_abs=min_thresh)
        peak_vals = pred[tuple(all_peaks.T)] if len(all_peaks) else np.empty(0)

        for thresh in tqdm(threshes):
            pred_coords = all_peaks[peak_vals >= thresh] if len(all_peaks) else all_peaks
            precision, recall, f1, _, _, _ = metric_coords(label_coords, pred_coords)

            data.append([val_path, thresh, f1, precision, recall])

    df = pandas.DataFrame(data=data, columns=["val_path", "Threshold", "f1", "precision", "recall"])

    # Aggregate over the whole validation set: average precision/recall per threshold.
    # (Previously idxmax picked a single (tomogram, threshold) row, i.e. the threshold
    # that happened to work best on the easiest single tomogram, not across the val set.)
    agg = df.groupby("Threshold", as_index=False)[["precision", "recall"]].mean()

    # F_beta with beta < 1 favours precision (fewer false positives / less oversampling).
    beta = GRIDSEARCH_BETA
    agg["f_beta"] = (
        (1 + beta**2) * agg["precision"] * agg["recall"]
        / (beta**2 * agg["precision"] + agg["recall"] + 1e-8)
    )
    best_thresh = float(agg.loc[agg["f_beta"].idxmax(), "Threshold"])

    print(f"per-threshold val averages (beta={beta}):")
    print(agg.to_string(index=False))
    print(f"The best threshold according to the val set {val_list} is {best_thresh}")

    return best_thresh