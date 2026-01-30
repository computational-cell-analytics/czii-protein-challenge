import argparse
import os
import zarr
from tqdm import tqdm

import numpy as np
import json

from detection.utils.prediction.prediction import get_prediction_torch_em
from detection.utils.inference.protein_detection import protein_detection
from detection.utils.training.tiling_helper import parse_tiling


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


def run_protein_detection(input_path, output_path, model_path, json_val_path, threshold=None):

    tiling = parse_tiling(tile_shape=None, halo=None) #TODO implement tiling and halo choices
    print(f"using tiling {tiling}")

    input_volume = get_volume(input_path)

    pred = get_prediction_torch_em(input_volume=input_volume, tiling=tiling, model_path=model_path, verbose=True)
    print(f"using the validation set listed in {json_val_path}")
    detections, threshold = protein_detection(pred, json_val_path, model_path, threshold=threshold)

    print(f"these are the results: {detections}")

    input_name = os.path.basename(input_path)
    output_folder = output_path
    os.makedirs(output_folder, exist_ok=True)

    '''#save prediction
    output_np_file = os.path.join(output_folder, f"{input_name}_protein_detections.npy")
    np.save(output_np_file, pred)
    print(f"Heatmap saved to {output_np_file}")'''

    # Save results to a JSON file
    output_json_file = os.path.join(output_folder, f"{input_name}_protein_detections.json")

    with open(output_json_file, "w") as f:
        json.dump(detections, f, indent=4)
    print(f"Coordinates saved to {output_json_file}")

    return threshold


def process_folder(args):
    input_files = []
    input_files = [os.path.join(args.input_path, name) for name in os.listdir(args.input_path)
                   if os.path.isdir(os.path.join(args.input_path, name))]

    threshold = None  # start with None, first run computes it

    pbar = tqdm(input_files, desc="Run protein detection")
    for input_path in pbar:
        input_name = os.path.basename(input_path)
        output_json = os.path.join(args.output_path, f"{input_name}_protein_detections.json")

        # Skip if already processed
        if os.path.exists(output_json):
            print(f"Skipping {input_name} - results already exist.")
            continue

        threshold = run_protein_detection(
            input_path, args.output_path, args.model_path, args.json_val_path, threshold=threshold
        )


def main():
    parser = argparse.ArgumentParser(description="Segment vesicles in EM tomograms.")
    parser.add_argument(
        "--input_path", "-i", required=True,
        help="The filepath to the mrc file or the directory containing the tomogram data."
    )
    parser.add_argument(
        "--file", "-f", action="store_true",
        help="Input path is a single file. The protein detection only needs to run once. By default, multiple files are expected."
    )
    parser.add_argument(
        "--output_path", "-o", required=True,
        help="The filepath to directory where the output will be saved."
    )
    parser.add_argument(
        "--model_path", "-m", required=True, help="The filepath to the model."
    )
    parser.add_argument(
        "--json_val_path", "-j", required=True, help="The json filepath to the validation split from the training."
    )
    args = parser.parse_args()

    file = args.file

    if file:
        _ = run_protein_detection(args.input_path, args.output_path, args.model_path, args.json_val_path)
    else:
        process_folder(args)

    print("Finished detecting!")


if __name__ == "__main__":
    main()
