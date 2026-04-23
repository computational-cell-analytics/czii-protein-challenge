import pandas
import time
from tqdm import tqdm
import zarr
import json
import os
from skimage.feature import blob_log, peak_local_max
from ..evaluation.evaluation_metrics import metric_coords
from ..prediction.prediction import get_prediction_torch_em
from ..training.tiling_helper import parse_tiling
from detection.data_processing.create_heatmap import parse_json_files
import numpy as np

from detection.config import ADJ_FACTOR, CZII_SMALLEST_PROTEIN_SIZE

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

    threshes = np.arange(1.0, 2.5, 0.1)
    data = []

    # Load JSON from the file
    with open(json_val_path, "r") as file:
        json_data = json.load(file)

    # Extract the 'val' list
    val_list = json_data["val"]

    for val_path in val_list:

        image_path = get_full_image_path(json_val_path= json_val_path, val_path=val_path)
        label_path = get_full_label_path(json_val_path= json_val_path, val_path=val_path)
        print(f"double check image path {image_path} and label path {label_path}")

        tiling = parse_tiling(tile_shape=None, halo=None) #TODO implement tiling and halo choices

        input_volume = get_volume(image_path)
        pred = get_prediction_torch_em(input_volume=input_volume, tiling=tiling, model_path=model_path, verbose=True)[0]

        json_files = [
            os.path.join(label_path, f)
            for f in os.listdir(label_path)
            if f.endswith('.json') and f not in ('no_class.json', 'albumin.json', "actin.json", "mt.json")
        ]
        label_coords, _ = parse_json_files(json_files)

        for thresh in tqdm(threshes):

            adj_factor = ADJ_FACTOR

            pred_coords = peak_local_max(pred, min_distance=int(CZII_SMALLEST_PROTEIN_SIZE*adj_factor * 0.9), threshold_abs=thresh)
            _, _, f1, _, _, _ = metric_coords(label_coords, pred_coords) 

            data.append([f1, thresh])
            print(f"f1 and corresponding thresholds: {data}")

        #Alternative using list conprehension
        '''
        adj_factor = ADJ_FACTOR     
        data.extend([
        [metric_coords(label_coords, blob_log(pred, min_sigma=CZII_SMALLEST_PROTEIN_SIZE * adj_factor * 0.9, 
                                            max_sigma=109.02 * adj_factor * 1.1, 
                                            threshold=thresh))[2], thresh]
        for thresh in tqdm(threshes)
        ])'''

    df = pandas.DataFrame(data=data, columns=["f1", "Threshold"])
    best_thresh = df.loc[df["f1"].idxmax(), "Threshold"]

    print(f"The best threshold according to the val set {val_list} is {best_thresh}")

    return best_thresh