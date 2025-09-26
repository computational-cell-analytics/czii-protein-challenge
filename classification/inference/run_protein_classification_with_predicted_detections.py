import argparse
import os
import json
import h5py
import numpy as np
from tqdm import tqdm

from classification.data_processing import extract_subtomograms
from classification.utils import protein_classification
from classification.training import get_coords_and_targets
from detection.utils import metric_coords

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


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



def load_detection_predictions(pred_path):
    with open(pred_path, "r") as f:
        points = json.load(f)
    return np.array(points)


def match_preds_with_labels(preds, label_path):
    """
    Match detection predictions with GT labels.
    Returns: matched_coords, matched_labels, num_unmatched
    """
    # load GT coords + labels
    json_files = [os.path.join(label_path, f) for f in os.listdir(label_path) if f.endswith('.json')]
    from detection.data_processing.create_heatmap import parse_json_files
    label_coords, protein_types = parse_json_files(json_files)

    # match predictions to labels (like in metric_coords)
    matched_idx_pred, matched_idx_gt = match_predictions_to_labels(preds, label_coords)

    matched_coords = [preds[i] for i in matched_idx_pred]
    matched_labels = [protein_types[j] for j in matched_idx_gt]

    num_unmatched = len(preds) - len(matched_coords)

    return matched_coords, matched_labels, num_unmatched


def extract_matched_subtomograms(volume, matched_coords, labels, max_extent, subtomo_output, tomo_name):
    os.makedirs(subtomo_output, exist_ok=True)
    file_paths, new_labels = [], []

    subtomograms, valid_coords, targets = extract_subtomograms(volume, matched_coords, max_extent, targets=labels)

    for cube, (x, y, z), target in zip(subtomograms, valid_coords, targets):
        filename = f"{tomo_name}_x{x}_y{y}_z{z}.h5"
        filepath = os.path.join(subtomo_output, filename)
        with h5py.File(filepath, "w") as f:
            f.create_dataset("raw", data=cube, compression="lzf")
        file_paths.append(filepath)
        new_labels.append(target)

    return file_paths, new_labels


def run_classification(subtomo_files, labels, model_path, batch_size=16):
    with open("/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/training/protein_classification_czii_v3/idx_to_label.json", "r") as f:
        idx_to_label = json.load(f)

    all_preds, all_probs = [], []
    all_truths, all_ids = [], []

    for i in range(0, len(subtomo_files), batch_size):
        batch_paths = subtomo_files[i:i + batch_size]
        cubes, sample_ids, truths = [], [], []

        for path, label in zip(batch_paths, labels[i:i + batch_size]):
            with h5py.File(path, "r") as f:
                cubes.append(f["raw"][:])
            sample_ids.append(os.path.basename(path))
            truths.append(label)

        cubes = np.stack(cubes, axis=0)
        probs, preds = protein_classification(cubes, model_path, EfficientNet=False)

        all_ids.extend(sample_ids)
        all_preds.extend([idx_to_label[str(p)] for p in preds])
        all_probs.extend(probs.tolist())
        all_truths.extend(truths)

    return all_ids, all_truths, all_preds, all_probs, idx_to_label


def evaluate_classification(ids, truths, preds, output_path, idx_to_label, num_unmatched, name="combined"):
    os.makedirs(output_path, exist_ok=True)

    cm = confusion_matrix(truths, preds, labels=list(idx_to_label.values()))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=idx_to_label.values(), yticklabels=idx_to_label.values(), cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(output_path, f"confusion_matrix_{name}.png"))
    plt.close()

    results_df = pd.DataFrame({
        "sample_id": ids,
        "truth": truths,
        "prediction": preds
    })
    results_df.to_csv(os.path.join(output_path, f"classification_results_{name}.csv"), index=False)

    # Add summary CSV with unmatched count
    summary_path = os.path.join(output_path, f"summary_{name}.csv")
    write_header = not os.path.exists(summary_path)
    with open(summary_path, "a") as f:
        if write_header:
            f.write("tomo_name,total_preds,matched_preds,unmatched_preds\n")
        f.write(f"{name},{len(ids)+num_unmatched},{len(ids)},{num_unmatched}\n")

    report = classification_report(truths, preds, labels=list(idx_to_label.values()))
    with open(os.path.join(output_path, f"classification_report_{name}.txt"), "w") as f:
        f.write(report)


def main():
    parser = argparse.ArgumentParser(description="Combined detection+classification evaluation.")
    parser.add_argument("--pred_coords", "-p", required=True, help="Detection prediction json file.")
    parser.add_argument("--label_path", "-l", required=True, help="GT labels folder (with Picks jsons).")
    parser.add_argument("--model_path", "-m", required=True, help="Classification model path.")
    parser.add_argument("--tomo_volume", "-v", required=True, help="Tomogram volume (zarr or mrc).")
    parser.add_argument("--max_extent", type=int, required=True, help="Subtomogram size.")
    parser.add_argument("--subtomo_output", "-sub_o", required=True, help="Where to save subtomograms.")
    parser.add_argument("--output_path", "-o", required=True, help="Where to save evaluation results.")
    parser.add_argument("--batch_size", type=int, default=16)

    args = parser.parse_args()

    preds = load_detection_predictions(args.pred_coords)
    matched_coords, matched_labels, num_unmatched = match_preds_with_labels(preds, args.label_path)

    from classification.inference.run_protein_classification import get_volume
    volume = get_volume(args.tomo_volume, zarr_=True)

    tomo_name = os.path.basename(args.tomo_volume.rstrip("/"))
    subtomo_files, labels = extract_matched_subtomograms(volume, matched_coords, matched_labels, args.max_extent, args.subtomo_output, tomo_name)

    ids, truths, preds, probs, idx_to_label = run_classification(subtomo_files, labels, args.model_path, args.batch_size)
    evaluate_classification(ids, truths, preds, args.output_path, idx_to_label, num_unmatched, name=tomo_name)

    print("Finished combined evaluation!")


if __name__ == "__main__":
    main()
