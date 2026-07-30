import argparse
import os
import glob
import h5py
import zarr
from tqdm import tqdm
import numpy as np
import torch

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from classification.training import load_peaks, get_volume
from classification.data_processing import extract_subtomograms
from classification.utils import protein_classification


def _resolve_max_extent_and_halo(model_path, fallback_max_extent):
    """Match run_protein_classification.py: extract cut-outs with the SAME geometry
    the model was trained/evaluated with -- max_extent = patch_shape[0] from the
    checkpoint and halo=0 (no resize). Falls back to --max_extent only if the
    checkpoint has no patch_shape.
    """
    ckpt_path = model_path if model_path.endswith("best.pt") else os.path.join(model_path, "best.pt")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    patch_shape = checkpoint.get("init", {}).get("patch_shape")
    if patch_shape is not None:
        max_extent = patch_shape[0]
        print(f"Using max_extent={max_extent} from checkpoint patch_shape, halo=0 "
              f"(matching run_protein_classification.py).")
        return max_extent, 0
    print(f"patch_shape not in checkpoint; falling back to --max_extent={fallback_max_extent}, halo=0.")
    return fallback_max_extent, 0


def run_protein_classification(input_paths, output_path: str, model_path: str, batch_size: int = 16):
    """
    Classify subtomograms from a list of .h5 files or a single file.
    Saves:
      - classification_results.csv: Sample ID, Predicted Class, Confidence
      - cluster_plot.png: t-SNE projection of probability vectors colored by predicted class
    """
    os.makedirs(output_path, exist_ok=True)

    if isinstance(input_paths, str):
        input_paths = [input_paths]

    all_sample_ids = []
    all_preds = []
    all_probs = []

    for i in range(0, len(input_paths), batch_size):
        batch_paths = input_paths[i:i + batch_size]

        cubes = []
        sample_ids = []
        for path in batch_paths:
            with h5py.File(path, "r") as f:
                cubes.append(f["raw"][:])
            sample_ids.append(os.path.basename(path))
        cubes = np.stack(cubes, axis=0)

        # Run classification, returns probabilities and predictions
        probs, preds = protein_classification(cubes, model_path)

        all_sample_ids.extend(sample_ids)
        all_preds.extend(preds)
        all_probs.extend(probs)

    all_probs = np.array(all_probs)

    # Save predictions list
    df = pd.DataFrame({
        "Sample ID": all_sample_ids,
        "Predicted Class": all_preds,
        "Confidence": np.max(all_probs, axis=1)
    })
    df_path = os.path.join(output_path, "classification_results.csv")
    df.to_csv(df_path, index=False)

    # Create and save t-SNE cluster plot from probability vectors
    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(all_probs)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        reduced[:, 0], reduced[:, 1],
        c=all_preds, cmap="tab20", alpha=0.7
    )
    plt.colorbar(scatter, label="Predicted Class")
    plt.title("Protein Classification Clusters (t-SNE from Probabilities)")
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    cluster_plot_path = os.path.join(output_path, "cluster_plot.png")
    plt.savefig(cluster_plot_path, dpi=300, bbox_inches="tight")
    plt.close()


def preprocess_tomo(input_tomo: str, detection_folder: str, max_extent: int, subtomo_output: str, halo: int = 0):
    experiment_name = os.path.basename(input_tomo)

    peaks_path = os.path.join(detection_folder, f"{experiment_name}_protein_detections.json")
    coords = load_peaks(peaks_path)

    raw_volume = get_volume(input_tomo)

    # halo=0 to match run_protein_classification.py (which extracts max_extent=patch_shape[0]
    # with halo=0 and no resize). The old default halo=4 silently changed the cube size.
    subtomograms, valid_coords = extract_subtomograms(raw_volume, coords, max_extent, halo=halo)

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
        #TODO case where multiple big tomograms:
        '''input_tomograms = [
            os.path.join(args.input_path, name)
            for name in os.listdir(args.input_path)
            if os.path.isdir(os.path.join(args.input_path, name))
        ]'''
        input_tomograms = [args.input_path]

        # Derive the cut-out geometry from the checkpoint so it matches training /
        # run_protein_classification.py (max_extent = patch_shape[0], halo=0, no resize).
        max_extent, halo = _resolve_max_extent_and_halo(args.model_path, args.max_extent)

        input_files = []
        for input_tomo in input_tomograms:
            paths = preprocess_tomo(
                input_tomo, args.detection, max_extent, args.subtomo_output, halo=halo
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
        help="Fallback bbox size, used ONLY if the checkpoint has no patch_shape. Normally "
             "max_extent is read from the checkpoint (= patch_shape[0]) so it matches training "
             "and run_protein_classification.py; extraction always uses halo=0 and no resize."
    )
    parser.add_argument(
        "--subtomo_output", "-sub_o", type=str,
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
