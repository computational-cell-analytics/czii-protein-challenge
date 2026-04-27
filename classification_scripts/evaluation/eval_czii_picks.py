import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from detection.utils import metric_coords


def parse_json_files(json_files):
    coordinates = []
    protein_types = []

    for file in json_files:
        with open(file, 'r') as f:
            data = json.load(f)
            points = data.get("points", [])
            for point in points:
                loc = point.get("location", {})
                x, y, z = loc.get("x"), loc.get("y"), loc.get("z")
                if x is not None and y is not None and z is not None:
                    coordinates.append((z / 10, y / 10, x / 10))  # scale
                    protein_types.append(data.get("pickable_object_name", "unknown"))

    return np.array(coordinates), protein_types


def match_predictions_to_labels(preds, gts, match_distance=45.0):
    n, m = len(gts), len(preds)
    if n == 0 or m == 0:
        return [], []

    pairwise_distances = cdist(gts, preds, metric="euclidean")
    max_distance = pairwise_distances.max() if pairwise_distances.size > 0 else 1.0
    costs = -(pairwise_distances < match_distance).astype(float) - (max_distance - pairwise_distances) / max_distance

    gt_idx, pred_idx = linear_sum_assignment(costs)
    match_mask = pairwise_distances[gt_idx, pred_idx] < match_distance
    matched_gt_idx = gt_idx[match_mask].tolist()
    matched_pred_idx = pred_idx[match_mask].tolist()

    return matched_pred_idx, matched_gt_idx


#Classification
def run_full_evaluation(sample_ids, truth_labels, pred_labels, probs, output_path, name, idx_to_label):
    cm = confusion_matrix(truth_labels, pred_labels, labels=list(idx_to_label.values()))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=idx_to_label.values(), yticklabels=idx_to_label.values(), cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"confusion_matrix_{name}.png"))
    plt.close()

    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", xticklabels=idx_to_label.values(), yticklabels=idx_to_label.values(), cmap="Blues", vmin=0, vmax=1)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Normalized Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"confusion_matrix_normalized_{name}.png"))
    plt.close()

    results_df = pd.DataFrame({"sample_id": sample_ids, "truth": truth_labels, "prediction": pred_labels})
    results_df.to_csv(os.path.join(output_path, f"classification_results_{name}.csv"), index=False)

    if len(probs) > 1:
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

    report = classification_report(truth_labels, pred_labels, labels=list(idx_to_label.values()))
    with open(os.path.join(output_path, f"classification_report_{name}.txt"), "w") as f:
        f.write(report)


# Detection
def evaluate_per_protein_type(pred_coords, label_path, model_name, input_name):
    json_files = [os.path.join(label_path, f) for f in os.listdir(label_path) if f.endswith('.json')]
    label_coords, protein_types = parse_json_files(json_files)

    # Organize label_coords by protein type
    label_dict = {}
    for coord, p_type in zip(label_coords, protein_types):
        label_dict.setdefault(p_type, []).append(coord)

    results_folder = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_folder, exist_ok=True)
    csv_file = os.path.join(results_folder, f"evaluation_{model_name}.csv")
    write_header = not os.path.exists(csv_file)

    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(["input_name", "protein_type", "precision", "recall", "f1",
                             "dev_percentage", "sMAPE", "mae", "label_count", "pred_count"])

        for protein_type, label_coords_subset in label_dict.items():
            precision, recall, f1, dev_percentage, sMAPE, mae = metric_coords(label_coords_subset, pred_coords)
            writer.writerow([
                input_name, protein_type, precision, recall, f1, dev_percentage, sMAPE, mae,
                len(label_coords_subset), len(pred_coords)
            ])
    print(f"Per-protein detection metrics saved to {csv_file}")


def evaluate_detection(pred_coords, label_path, model_name, input_name, evaluate_per_protein=True):
    json_files = [os.path.join(label_path, f) for f in os.listdir(label_path) if f.endswith('.json')]
    label_coords, _ = parse_json_files(json_files)
    predictions = np.array(pred_coords)

    precision, recall, f1, dev_percentage, sMAPE, mae = metric_coords(label_coords, predictions)

    results_folder = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_folder, exist_ok=True)
    csv_file = os.path.join(results_folder, f"evaluation_{model_name}.csv")
    write_header = not os.path.exists(csv_file)

    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(["input_name", "protein_type", "precision", "recall", "f1",
                             "dev_percentage", "sMAPE", "mae", "label_count", "pred_count"])
        writer.writerow([
            input_name, "all", precision, recall, f1, dev_percentage, sMAPE, mae,
            len(label_coords), len(predictions)
        ])
    print(f"Overall detection metrics saved to {csv_file}")

    if evaluate_per_protein:
        evaluate_per_protein_type(predictions, label_path, model_name, input_name)


def evaluate_dataset(ground_truth_dir, predictions_dir, output_dir, match_distance=45.0):
    os.makedirs(output_dir, exist_ok=True)

    classes = ["apo-ferritin", "beta-amylase", "beta-galactosidase", "ribosome", "thyroglobulin", "virus-like-particle"]
    idx_to_label = {i: cls for i, cls in enumerate(classes)}

    all_sample_ids, all_truth_labels, all_pred_labels, all_probs = [], [], [], []

    for tomo_id in sorted(os.listdir(ground_truth_dir)):
        gt_tomo_path = os.path.join(ground_truth_dir, tomo_id, "Picks")
        pred_tomo_path = os.path.join(predictions_dir, tomo_id, "Picks")
        if not os.path.isdir(gt_tomo_path) or not os.path.isdir(pred_tomo_path):
            continue

        gt_jsons = [os.path.join(gt_tomo_path, f) for f in os.listdir(gt_tomo_path) if f.endswith(".json")]
        pred_jsons = [os.path.join(pred_tomo_path, f) for f in os.listdir(pred_tomo_path) if f.endswith(".json")]

        gt_coords, gt_labels = parse_json_files(gt_jsons)
        pred_coords, pred_labels = parse_json_files(pred_jsons)

        evaluate_detection(pred_coords, gt_tomo_path, model_name="1_Daddies", input_name=tomo_id)

        matched_pred_idx, matched_gt_idx = match_predictions_to_labels(pred_coords, gt_coords, match_distance=match_distance)

        # matched pairs
        for gt_i, pred_i in zip(matched_gt_idx, matched_pred_idx):
            all_sample_ids.append(f"{tomo_id}_{gt_i}")
            all_truth_labels.append(gt_labels[gt_i])
            all_pred_labels.append(pred_labels[pred_i])

            prob_vector = np.zeros(len(classes))
            prob_vector[classes.index(pred_labels[pred_i])] = 1.0 if pred_labels[pred_i] in classes else 1.0 / len(classes)
            all_probs.append(prob_vector)

    # Run classification evaluation only on matched points
    run_full_evaluation(all_sample_ids, all_truth_labels, all_pred_labels, all_probs, output_dir, name="public_test_dataset", idx_to_label=idx_to_label)


if __name__ == "__main__":
    ground_truth_dir = "/scratch-grete/projects/nim00007/cryo-et/challenge-data/public_test_dataset/ground_truth"
    predictions_dir = "/scratch-grete/projects/nim00007/cryo-et/challenge-data/1_Daddies"
    output_dir = "/mnt/ceph-hdd/cold/nim00007/czii/evaluation_czii_top3/1_Daddies"

    evaluate_dataset(ground_truth_dir, predictions_dir, output_dir, match_distance=45.0)
