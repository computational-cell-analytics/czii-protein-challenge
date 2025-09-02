import argparse
import os
import h5py
import zarr
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

from typing import List, Tuple

from classification.data_processing import extract_subtomograms
from classification.utils import protein_classification
from classification.training import get_coords_and_targets


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
        zarr_path = os.path.join(input_path, "VoxelSpacing10.000", "denoised.zarr", "0")
        zarr_file = zarr.open(zarr_path, mode="r")
        volume = zarr_file[:]
    else:
        volume = get_non_zarr(input_path)

    return volume



def preprocess_tomo_with_labels(
    tomo_paths: List[str],
    labels_root: str,
    max_extent: int,
    subtomo_output: str
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

        subtomograms, valid_coords, targets = extract_subtomograms(raw_volume, coords, max_extent, targets=targets)

        for cube, (x, y, z), target in zip(subtomograms, valid_coords, targets):
            filename = f"{experiment_name}_x{x}_y{y}_z{z}.h5"
            filepath = os.path.join(subtomo_output, filename)

            with h5py.File(filepath, "w") as f:
                f.create_dataset("raw", data=cube, compression="lzf")

            file_paths.append(filepath)
            labels.append(target)

    return file_paths, labels




def run_protein_classification_with_labels(
    input_files,
    labels,
    output_path,
    model_path,
    tomo_name,
    batch_size=16
):
    os.makedirs(output_path, exist_ok=True)

    # Load int→label mapping
    with open("/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/training/protein_classification_czii_v3/idx_to_label.json", "r") as f:
        idx_to_label = json.load(f)

    all_sample_ids, all_preds, all_probs, all_truths = [], [], [], []

    # Batch processing
    for i in range(0, len(input_files), batch_size):
        batch_paths = input_files[i:i + batch_size]
        cubes, sample_ids, truths_batch = [], [], []

        for path, label in zip(batch_paths, labels[i:i + batch_size]):
            with h5py.File(path, "r") as f:
                cubes.append(f["raw"][:])
            sample_ids.append(os.path.basename(path))
            truths_batch.append(label)

        cubes = np.stack(cubes, axis=0)
        probs, preds = protein_classification(cubes, model_path, EfficientNet=False)

        all_sample_ids.extend(sample_ids)
        all_preds.extend(preds)
        all_probs.extend(probs.tolist())
        all_truths.extend(truths_batch)

    # --- Convert int preds to labels ---
    all_pred_labels = [idx_to_label[str(p)] for p in all_preds]
    all_truth_labels = [idx_to_label[str(t)] if str(t) in idx_to_label else t for t in all_truths]

    # --- Save confusion matrix ---
    cm = confusion_matrix(all_truth_labels, all_pred_labels, labels=list(idx_to_label.values()))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=idx_to_label.values(), yticklabels=idx_to_label.values(), cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"confusion_matrix_{tomo_name}.png"))
    plt.close()

    # --- Save results as list (CSV) ---
    results_df = pd.DataFrame({
        "sample_id": all_sample_ids,
        "truth": all_truth_labels,
        "prediction": all_pred_labels
    })
    results_df.to_csv(os.path.join(output_path, f"classification_results_{tomo_name}.csv"), index=False)

    # --- Save scatter plot of embeddings (t-SNE of probs) ---
    tsne = TSNE(n_components=2, random_state=42)
    probs_2d = tsne.fit_transform(np.array(all_probs))

    plt.figure(figsize=(8, 6))
    for label in set(all_truth_labels):
        mask = np.array(all_truth_labels) == label
        plt.scatter(probs_2d[mask, 0], probs_2d[mask, 1], label=label, alpha=0.6)

    plt.legend()
    plt.title("t-SNE of classification probabilities")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"tsne_scatter_{tomo_name}.png"))
    plt.close()

    # --- Save classification report ---
    report = classification_report(all_truth_labels, all_pred_labels, labels=list(idx_to_label.values()))
    with open(os.path.join(output_path, f"classification_report_{tomo_name}.txt"), "w") as f:
        f.write(report)



def main():
    parser = argparse.ArgumentParser(description="Protein classification with ground truth labels.")
    parser.add_argument(
        "--input_path", "-i", required=True, type=str,
        help="Path to tomogram directory or single tomogram folder."
    )
    parser.add_argument(
        "--labels_root", "-l", required=True, type=str,
        help="Root directory containing label JSON files."
    )
    parser.add_argument(
        "--output_path", "-o", required=True, type=str,
        help="Directory where results will be saved."
    )
    parser.add_argument(
        "--model_path", "-m", required=True, type=str,
        help="Path to trained model."
    )
    parser.add_argument(
        "--max_extent", type=int, required=True,
        help="Size of the bounding box used during training."
    )
    parser.add_argument(
        "--subtomo_output", "-sub_o", required=True, type=str,
        help="Where to store intermediate subtomogram .h5 files."
    )
    parser.add_argument(
        "--batch_size", "-b", type=int, default=16,
        help="Batch size for classification."
    )
    parser.add_argument(
        "--multiple", "-mlp", action="store_true",
        help="Activate when input_path contains multiple directories for inference."
    )


    args = parser.parse_args()
    
    if args.multiple:
        # Iterate over all subfolders in the input path
        for subfolder in os.listdir(args.input_path):
            subfolder_path = os.path.join(args.input_path, subfolder)
            if os.path.isdir(subfolder_path):
                tomo_paths = [subfolder_path]

                print(f"Extracting subtomograms with labels from {subfolder}...")
                subtomo_files, labels = preprocess_tomo_with_labels(
                    tomo_paths,
                    args.labels_root,
                    args.max_extent,
                    args.subtomo_output
                )

                tomo_name = os.path.basename(subfolder_path)

                print(f"Classifying {len(subtomo_files)} subtomograms from {subfolder}...")
                run_protein_classification_with_labels(
                    subtomo_files,
                    labels,
                    args.output_path,
                    args.model_path,
                    tomo_name=tomo_name,
                    batch_size=args.batch_size
                )
    else:
        tomo_paths = [args.input_path]

        print("Extracting subtomograms with labels...")
        subtomo_files, labels = preprocess_tomo_with_labels(
            tomo_paths,
            args.labels_root,
            args.max_extent,
            args.subtomo_output
        )

        tomo_name = os.path.basename(args.input_path)

        print(f"Classifying {len(subtomo_files)} subtomograms...")
        run_protein_classification_with_labels(
            subtomo_files,
            labels,
            args.output_path,
            args.model_path,
            tomo_name=tomo_name,
            batch_size=args.batch_size
        )

    print("Finished classification with label evaluation!")


if __name__ == "__main__":
    main()
