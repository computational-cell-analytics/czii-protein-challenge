import argparse
import os
import h5py
import zarr
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import random

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


def run_full_evaluation(sample_ids, truth_labels, pred_labels, probs, output_path, name, idx_to_label):
    # Save confusion matrix (raw counts)
    cm = confusion_matrix(truth_labels, pred_labels, labels=list(idx_to_label.values()))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=idx_to_label.values(), yticklabels=idx_to_label.values(), cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"confusion_matrix_{name}.png"))
    plt.close()

    # Save confusion matrix (normalized: 0-1)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", xticklabels=idx_to_label.values(), yticklabels=idx_to_label.values(), cmap="Blues", vmin=0, vmax=1)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"confusion_matrix_normalized_{name}.png"))
    plt.close()
    '''
    #Save results as list (CSV) 
    results_df = pd.DataFrame({
        "sample_id": sample_ids,
        "truth": truth_labels,
        "prediction": pred_labels
    })
    results_df.to_csv(os.path.join(output_path, f"classification_results_{name}.csv"), index=False)

    # Save scatter plot of embeddings (t-SNE of probs)
    tsne = TSNE(n_components=2, random_state=42)
    probs_2d = tsne.fit_transform(np.array(probs))

    plt.figure(figsize=(8, 6))
    for label in set(truth_labels):
        mask = np.array(truth_labels) == label
        plt.scatter(probs_2d[mask, 0], probs_2d[mask, 1], label=label, alpha=0.6)

    plt.legend()
    plt.title(f"t-SNE of classification probabilities - {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"tsne_scatter_{name}.png"))
    plt.close()
    '''
    # Save classification report
    report = classification_report(truth_labels, pred_labels, labels=list(idx_to_label.values()))
    with open(os.path.join(output_path, f"classification_report_{name}.txt"), "w") as f:
        f.write(report)


def run_global_evaluation(global_ids, global_truths, global_preds, global_probs, output_path, idx_to_label):
    run_full_evaluation(global_ids, global_truths, global_preds, global_probs, output_path, "ALL", idx_to_label)


def run_protein_classification_with_labels(
    input_files,
    labels,
    output_path,
    model_path,
    tomo_name,
    batch_size=16,
    save_full_results=True
):
    os.makedirs(output_path, exist_ok=True)

    # Load int to label mapping
    with open("/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/training/protein_classification_czii_v21/idx_to_label.json", "r") as f:
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

    # Convert int preds to labels
    all_pred_labels = [idx_to_label[str(p)] for p in all_preds]
    all_truth_labels = [idx_to_label[str(t)] if str(t) in idx_to_label else t for t in all_truths]

    if save_full_results:
        # Full evaluation on all subtomograms
        run_full_evaluation(all_sample_ids, all_truth_labels, all_pred_labels, all_probs, output_path, tomo_name, idx_to_label)

    return all_sample_ids, all_truth_labels, all_pred_labels, all_probs


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
        global_ids, global_truths, global_preds, global_probs = [], [], [], []

        # Collect all tomogram subfolders
        tomogram_folders = [os.path.join(args.input_path, sf) for sf in os.listdir(args.input_path) if os.path.isdir(os.path.join(args.input_path, sf))]

        # Pick 10% of tomograms for per-tomogram evaluation
        n_eval = max(1, int(len(tomogram_folders) * 0.1))
        eval_tomos = set(random.sample(tomogram_folders, n_eval))

        for subfolder_path in tomogram_folders:
            tomo_paths = [subfolder_path]

            print(f"Extracting subtomograms with labels from {subfolder_path}...")
            subtomo_files, labels = preprocess_tomo_with_labels(
                tomo_paths,
                args.labels_root,
                args.max_extent,
                args.subtomo_output
            )

            tomo_name = os.path.basename(subfolder_path)

            print(f"Classifying {len(subtomo_files)} subtomograms from {subfolder_path}...")
            sample_ids, truths, preds, probs = run_protein_classification_with_labels(
                subtomo_files,
                labels,
                args.output_path,
                args.model_path,
                tomo_name=tomo_name,
                batch_size=args.batch_size,
                save_full_results=(subfolder_path in eval_tomos)  # only full eval for 10% tomograms
            )

            global_ids.extend(sample_ids)
            global_truths.extend(truths)
            global_preds.extend(preds)
            global_probs.extend(probs)

        # Global evaluation
        with open("/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/training/protein_classification_czii_v21/idx_to_label.json", "r") as f:
            idx_to_label = json.load(f)
        run_global_evaluation(global_ids, global_truths, global_preds, global_probs, args.output_path, idx_to_label)

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
            batch_size=args.batch_size,
            save_full_results=True  # full evaluation in single mode
        )

    print("Finished classification with label evaluation!")


if __name__ == "__main__":
    main()
