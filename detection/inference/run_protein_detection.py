import argparse
import os
import zarr
from tqdm import tqdm

import numpy as np
import json

from detection.utils.prediction.prediction import get_prediction_torch_em
from detection.utils.inference.protein_detection import protein_detection
from detection.utils.training.tiling_helper import parse_tiling

def get_volume(input_path):
    zarr_file = zarr.open(os.path.join(input_path, "VoxelSpacing10.000", "denoised.zarr", "0"), mode='r')
    input_volume = zarr_file[:]

    return input_volume

def run_protein_detection(input_path, output_path, model_path, json_val_path):

    tiling = parse_tiling(tile_shape=None, halo=None) #TODO implement tiling and halo choices
    print(f"using tiling {tiling}")

    input_volume = get_volume(input_path)

    pred = get_prediction_torch_em(input_volume=input_volume, tiling=tiling, model_path=model_path, verbose=True)
    print(f"using the validation set listed in {json_val_path}")
    detections = protein_detection(pred, json_val_path, model_path)

    print(f"these are the results: {detections}")

    model_name = os.path.basename(os.path.normpath(model_path))
    input_name = os.path.basename(input_path)
    output_folder = os.path.join(output_path, model_name)
    os.makedirs(output_folder, exist_ok=True)

    #save prediction
    output_np_file = os.path.join(output_folder, f"{input_name}_protein_detections.npy")
    np.save(output_np_file, pred)
    print(f"Heatmap saved to {output_np_file}")

    # Save results to a JSON file
    output_json_file = os.path.join(output_folder, f"{input_name}_protein_detections.json")

    with open(output_json_file, "w") as f:
        json.dump(detections, f, indent=4)
    print(f"Coordinates saved to {output_json_file}")


def process_folder(args):
    input_files = []
    input_files = [os.path.join(args.input_path, name) for name in os.listdir(args.input_path)
                   if os.path.isdir(os.path.join(args.input_path, name))]

    pbar = tqdm(input_files, desc="Run protein detection")
    for input_path in pbar:

        run_protein_detection(
            input_path, args.output_path, args.model_path, args.json_val_path
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
        run_protein_detection(args.input_path, args.output_path, args.model_path, args.json_val_path)
    else:
        process_folder(args)


    print("Finished segmenting!")

if __name__ == "__main__":
    main()
