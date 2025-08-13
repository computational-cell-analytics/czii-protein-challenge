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

#TODO Do I want to make this more flexible??
TRAIN_ROOT = "/scratch-grete/projects/nim00007/cryo-et/challenge-data/train/static/"
LABEL_ROOT = "/scratch-grete/projects/nim00007/cryo-et/challenge-data/train/overlay/"


def get_volume(input_path):
    zarr_file = zarr.open(os.path.join(input_path, "VoxelSpacing10.000", "denoised.zarr", "0"), mode='r')
    input_volume = zarr_file[:]

    return input_volume

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

    label_path = os.path.join(LABEL_ROOT, experiment_name, val_path,"Picks")

    return label_path

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

        image_path = get_full_image_path(json_val_path, val_path)
        label_path = get_full_label_path(json_val_path, val_path)

        tiling = parse_tiling(tile_shape=None, halo=None) #TODO implement tiling and halo choices

        input_volume = get_volume(image_path)
        pred = get_prediction_torch_em(input_volume=input_volume, tiling=tiling, model_path=model_path, verbose=True)[0]

        
        json_files = [os.path.join(label_path, f) for f in os.listdir(label_path) if f.endswith('.json')]
        label_coords, _ = parse_json_files(json_files)



        for thresh in tqdm(threshes):

            #smalles protein structure: "beta-amylase": 33.27
            #bigges protein structure: "ribosome": 109.02,
            #0.3 is the factor to match the PDB size to the experimental data size
            adj_factor=0.3 #TODO implement this as an argument, also when creating heatmap

            #TODO decide on blob_log or peak_local_max; blob_log is SUPER slow
            '''# Start timing
            start_time = time.time()

            pred_coords_sigma = blob_log(pred, min_sigma=33.27*adj_factor *0.9, max_sigma=109.02*adj_factor*1.1, threshold=thresh) 
            pred_coords = pred_coords_sigma[:, :-1]  # This removes the last column (sigma)

            # Stop timing
            elapsed_time = time.time() - start_time
            print(f"blob_log took {elapsed_time:.4f} seconds")
            '''
            pred_coords = peak_local_max(pred, min_distance=int(33.27*adj_factor *0.9), threshold_abs=thresh)
            _, _, f1, _, _, _ = metric_coords(label_coords, pred_coords) 

            data.append([f1, thresh])
            print(f"f1 and corresponding thresholds: {data}")

        #Alternative using list conprehension
        '''#smalles protein structure: "beta-amylase": 33.27
        #bigges protein structure: "ribosome": 109.02,
        #0.3 is the factor to match the PDB size to the experimental data size
        adj_factor=0.3 #TODO implement this as an argument, also when creating heatmap      
        data.extend([
        [metric_coords(label_coords, blob_log(pred, min_sigma=33.27 * adj_factor * 0.9, 
                                            max_sigma=109.02 * adj_factor * 1.1, 
                                            threshold=thresh))[2], thresh]
        for thresh in tqdm(threshes)
        ])'''

    df = pandas.DataFrame(data=data, columns=["f1", "Threshold"])
    best_thresh = df.loc[df["f1"].idxmax(), "Threshold"]

    print(f"The best threshold according to the val set {val_list} is {best_thresh}")

    return best_thresh