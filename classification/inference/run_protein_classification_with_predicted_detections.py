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
import random

from typing import List, Tuple
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from classification.data_processing import extract_subtomograms
from classification.utils import protein_classification
from classification.training import get_coords_and_targets

def load_detection_predictions(pred_path):
    with open(pred_path, "r") as f:
        points = json.load(f)
    return np.array(points)

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

def match_predictions_to_labels(preds, gts, match_distance=45):
    """
    Match predicted coordinates to ground-truth coordinates using Hungarian algorithm.
    
    Args:
        preds: np.ndarray of shape (M, D) - predicted coordinates
        gts: np.ndarray of shape (N, D) - ground-truth coordinates
        match_distance: float, maximum distance to consider a match.
    
    Returns:
        matched_pred_idx: list of indices in preds that were matched
        matched_gt_idx: list of indices in gts that were matched
    """
    n, m = len(gts), len(preds)

    if n == 0 or m == 0:
        return [], []

    # Compute pairwise distances
    pairwise_distances = cdist(gts, preds, metric="euclidean")

    # Define costs (Hungarian finds *minimum* cost)
    max_distance = pairwise_distances.max() if pairwise_distances.size > 0 else 1.0
    costs = -(pairwise_distances < match_distance).astype(float) - \
            (max_distance - pairwise_distances) / max_distance

    # Hungarian matching
    gt_idx, pred_idx = linear_sum_assignment(costs)

    # Keep only valid matches under distance threshold
    match_mask = pairwise_distances[gt_idx, pred_idx] < match_distance
    matched_gt_idx = gt_idx[match_mask].tolist()
    matched_pred_idx = pred_idx[match_mask].tolist()

    return matched_pred_idx, matched_gt_idx

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
    # Load and parse JSONs, skipping albumin.json
    json_files = [
        os.path.join(label_path, f)
        for f in os.listdir(label_path)
        if f.endswith('.json') and f != "albumin.json" #need this for synthetic data
    ]
    from detection.data_processing.create_heatmap import parse_json_files
    label_coords, protein_types = parse_json_files(json_files)
    #TODO do I need this?
    '''
    # Convert to integers for array slicing
    label_coords = [(int(round(x)), int(round(y)), int(round(z))) for x, y, z in label_coords]
    '''
    label_coords = np.array(label_coords)
    protein_types = np.array(protein_types)

    #Remove GT points labeled no_class (they cannot match)
    keep_mask = protein_types != no_class_label
    real_gt_coords = label_coords[keep_mask]
    real_gt_labels = protein_types[keep_mask]

    #Run matching 
    matched_idx_pred, matched_idx_gt = match_predictions_to_labels(preds, real_gt_coords)

    #create arrays for final output
    num_preds = len(preds)
    assigned_labels = [no_class_label] * num_preds   # default label for all predictions
    matched_coords = [preds[i] for i in range(num_preds)]  # keep all preds
    matched_coords = np.round(matched_coords).astype(int)

    # assign matched labels
    for p_idx, gt_idx in zip(matched_idx_pred, matched_idx_gt):
        assigned_labels[p_idx] = real_gt_labels[gt_idx]


    #Count predictions that were not matched
    num_unmatched_true = sum(1 for lbl in assigned_labels if lbl == no_class_label)

    return matched_coords, assigned_labels, num_unmatched_true

def preprocess_tomo_with_predictions(
    tomo_paths: List[str],
    pred_path: str,
    label_path: str,
    max_extent: int,
    subtomo_output: str
) -> Tuple[List[str], List[str]]:
    """
    Extract subtomograms at labeled coordinates and save them as .h5 files.

    Returns:
        file_paths: list of .h5 file paths
        labels: list of corresponding ground-truth labels (one per file)
    """
    # Load predictions
    preds = load_detection_predictions(pred_path)
    # Match to GT
    coords, targets, num_unmatched = match_preds_with_labels(preds, label_path)

    os.makedirs(subtomo_output, exist_ok=True)

    file_paths = []
    labels = []

    for tomo_path in tomo_paths:
        experiment_name = os.path.basename(tomo_path)
        raw_volume = get_volume(tomo_path, zarr_=True)
        coords = coords.astype(int)
        # Convert from (x, y, z) to (z, y, x)
        coords = [(int(c[2]), int(c[1]), int(c[0])) for c in coords]

        subtomograms, valid_coords, targets = extract_subtomograms(raw_volume, coords, max_extent, targets=targets)

        for cube, (x, y, z), target in zip(subtomograms, valid_coords, targets):
            filename = f"{experiment_name}_x{x}_y{y}_z{z}.h5"
            filepath = os.path.join(subtomo_output, filename)

            with h5py.File(filepath, "w") as f:
                f.create_dataset("raw", data=cube, compression="lzf")

            file_paths.append(filepath)
            labels.append(target)

    return file_paths, labels, num_unmatched


def run_full_evaluation(sample_ids, truth_labels, pred_labels, probs, output_path, name, idx_to_label, num_unmatched):
    # --- Save confusion matrix (raw counts) ---
    cm = confusion_matrix(truth_labels, pred_labels, labels=list(idx_to_label.values()))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=idx_to_label.values(), yticklabels=idx_to_label.values(), cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"confusion_matrix_{name}.png"))
    plt.close()

    # --- Save confusion matrix (normalized: 0-1) ---
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
    # --- Save results as list (CSV) ---
    results_df = pd.DataFrame({
        "sample_id": sample_ids,
        "truth": truth_labels,
        "prediction": pred_labels
    })
    results_df.to_csv(os.path.join(output_path, f"classification_results_{name}.csv"), index=False)

    # --- Save scatter plot of embeddings (t-SNE of probs) ---
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
    # --- Save classification report ---
    report = classification_report(truth_labels, pred_labels, labels=list(idx_to_label.values()))
    with open(os.path.join(output_path, f"classification_report_{name}.txt"), "w") as f:
        f.write(report)
    
    #Write summary including unmatched predictions (= no_class)
    summary_path = os.path.join(output_path, f"summary_{name}.csv")
    write_header = not os.path.exists(summary_path)
    with open(summary_path, "a") as f:
        if write_header:
            f.write("tomo_name,total_preds,matched_preds,unmatched_preds\n")
        f.write(f"{name},{len(sample_ids)+num_unmatched},{len(sample_ids)},{num_unmatched}\n")



def run_global_evaluation(global_ids, global_truths, global_preds, global_probs, output_path, idx_to_label, num_unmatched):
    run_full_evaluation(global_ids, global_truths, global_preds, global_probs, output_path, "ALL", idx_to_label, num_unmatched)


def run_protein_classification_with_labels(
    input_files,
    labels,
    output_path,
    model_path,
    tomo_name,
    batch_size=16,
    save_full_results=True,
    num_unmatched=None
):
    os.makedirs(output_path, exist_ok=True)

    # Load int→label mapping
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

    # --- Convert int preds to labels ---
    all_pred_labels = [idx_to_label[str(p)] for p in all_preds]
    all_truth_labels = [idx_to_label[str(t)] if str(t) in idx_to_label else t for t in all_truths]

    if save_full_results:
        # Full evaluation on all subtomograms
        run_full_evaluation(all_sample_ids, all_truth_labels, all_pred_labels, all_probs, output_path, tomo_name, idx_to_label, num_unmatched)

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
        "--pred_coords", "-p", required=True,
        help="Either a single detection JSON file or a directory of files."
    )

    args = parser.parse_args()
    
    if os.path.isdir(args.pred_coords):
        global_ids, global_truths, global_preds, global_probs = [], [], [], []

        # Collect all tomogram subfolders
        tomogram_folders = [os.path.join(args.input_path, sf) for sf in os.listdir(args.input_path) if os.path.isdir(os.path.join(args.input_path, sf))]

        # Pick 10% of tomograms for per-tomogram evaluation
        n_eval = max(1, int(len(tomogram_folders) * 0.1))
        eval_tomos = set(random.sample(tomogram_folders, n_eval))

        global_num_unmatched=0

        for subfolder_path in tomogram_folders:
            tomo_paths = [subfolder_path]
            tomoID = os.path.basename(subfolder_path)

            print(f"Extracting subtomograms with labels from {subfolder_path}...")
            subtomo_files, labels, num_unmatched = preprocess_tomo_with_predictions(
                tomo_paths,
                pred_path = os.path.join(args.pred_coords, f"{tomoID}_protein_detections.json"),
                label_path=os.path.join(args.labels_root, tomoID, "Picks"),
                max_extent=args.max_extent,
                subtomo_output=args.subtomo_output
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
                save_full_results=(subfolder_path in eval_tomos),  # only full eval for 10% tomograms
                num_unmatched= num_unmatched
            )

            global_ids.extend(sample_ids)
            global_truths.extend(truths)
            global_preds.extend(preds)
            global_probs.extend(probs)
            global_num_unmatched+=num_unmatched

        # --- Global evaluation ---
        with open("/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/training/protein_classification_czii_v21/idx_to_label.json", "r") as f:
            idx_to_label = json.load(f)
        run_global_evaluation(global_ids, global_truths, global_preds, global_probs, args.output_path, idx_to_label, global_num_unmatched)

    else:
        tomo_paths = [args.input_path]
        tomoID = os.path.basename(args.input_path)

        print("Extracting subtomograms with labels...")
        subtomo_files, labels, num_unmatched = preprocess_tomo_with_predictions(
            tomo_paths,
            pred_path = os.path.join(args.pred_coords, f"{tomoID}_protein_detections.json"),
            label_path=os.path.join(args.labels_root, "Picks"),
            max_extent=args.max_extent,
            subtomo_output=args.subtomo_output
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
            save_full_results=True,  # full evaluation in single mode,
            num_unmatched=num_unmatched
        )

    print("Finished classification with label evaluation!")


if __name__ == "__main__":
    main()
