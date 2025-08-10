import argparse
import os
import glob
import h5py
import zarr
from tqdm import tqdm
import numpy as np
import json

from classification.training import load_heatmap, load_peaks
from classification.data_processing import extract_subtomograms
from classification.utils.inference import protein_classification


def get_volume(input_path: str) -> np.ndarray:
    zarr_file = zarr.open(os.path.join(input_path, "VoxelSpacing10.000", "denoised.zarr", "0"), mode='r')
    input_volume = zarr_file[:]
    return input_volume


def run_protein_classification(input_paths, output_path: str, model_path: str, batch_size: int = 16):
    """
    Classify subtomograms from a list of .h5 files or a single file.
    Saves results alongside predictions and probabilities.
    """
    os.makedirs(output_path, exist_ok=True)

    if isinstance(input_paths, str):
        input_paths = [input_paths]

    results_paths = []
    for i in range(0, len(input_paths), batch_size):
        batch_paths = input_paths[i:i + batch_size]

        cubes = []
        for path in batch_paths:
            with h5py.File(path, "r") as f:
                cubes.append(f["raw"][:])
        cubes = np.stack(cubes, axis=0)

        # Run classification
        probs, preds = protein_classification(cubes, model_path)

        # Save results for each file
        for path, prob, pred in zip(batch_paths, probs, preds):
            base_name = os.path.splitext(os.path.basename(path))[0]
            out_file = os.path.join(output_path, f"{base_name}_classification.h5")
            with h5py.File(out_file, "w") as f:
                f.create_dataset("probabilities", data=prob)
                f.create_dataset("predicted_class", data=np.array(pred, dtype=np.int64))
            results_paths.append(out_file)

    return results_paths


def preprocess_tomo(input_tomo: str, detection_folder: str, max_extent: int, subtomo_output: str):
    experiment_name = os.path.splitext(os.path.basename(input_tomo))[0]

    peaks_path = os.path.join(detection_folder, f"{experiment_name}_protein_detections.json")
    coords = load_peaks(peaks_path)

    raw_volume = get_volume(input_tomo)

    subtomograms, valid_coords = extract_subtomograms(raw_volume, coords, max_extent)

    file_paths = []
    os.makedirs(subtomo_output, exist_ok=True)

    for cube, (z, y, x) in zip(subtomograms, valid_coords):
        filename = f"{experiment_name}_z{z}_y{y}_x{x}.h5"
        filepath = os.path.join(subtomo_output, filename)

        with h5py.File(filepath, "w") as f:
            f.create_dataset("raw", data=cube, compression="lzf")

        file_paths.append(filepath)

    return file_paths


def process_folder(args):
    if args.preprocess:
        input_tomograms = [
            os.path.join(args.input_path, name)
            for name in os.listdir(args.input_path)
            if os.path.isdir(os.path.join(args.input_path, name))
        ]

        input_files = []
        for input_tomo in input_tomograms:
            paths = preprocess_tomo(
                input_tomo, args.detection, args.max_extent, args.subtomo_output
            )
            input_files.extend(paths)  # Flatten list
    else:
        input_files = glob.glob(os.path.join(args.input_path, '**', '*.h5'), recursive=True)

    tqdm.write(f"Found {len(input_files)} subtomograms to classify.")
    run_protein_classification(input_files, args.output_path, args.model_path)


def main():
    parser = argparse.ArgumentParser(description="Segment vesicles in EM tomograms.")
    parser.add_argument(
        "--input_path", "-i", required=True, type=str,
        help="The filepath to the zarr file or the directory containing the tomogram data."
    )
    parser.add_argument(
        "--multiple", "-mpl", action="store_true",
        help="Input path is a directory of multiple files or a full tomogram that is used to extract MULTIPLE subtomograms."
    )
    parser.add_argument(
        "--output_path", "-o", required=True, type=str,
        help="The filepath to directory where the output will be saved."
    )
    parser.add_argument(
        "--model_path", "-m", required=True, type=str,
        help="The filepath to the model."
    )
    parser.add_argument(
        "--preprocess", action="store_true",
        help="Set to true if the input is a tomogram, not the subtomograms, and the tomogram needs preprocessing before the classification."
    )
    parser.add_argument(
        "--detection", "-d", type=str,
        help="If the input is a full tomogram, not the subtomograms, the directory of the heatmap and the lists of the coordinates is needed."
    )
    parser.add_argument(
        "--max_extent", type=int,
        help="Size of the bbox that was used during training"
    )
    parser.add_argument(
        "--subtomo_output", type=str,
        help="Where should the subtomograms be stored"
    )

    args = parser.parse_args()

    if args.multiple:
        process_folder(args)
    else:
        run_protein_classification(args.input_path, args.output_path, args.model_path)

    print("Finished classification!")


if __name__ == "__main__":
    main()
