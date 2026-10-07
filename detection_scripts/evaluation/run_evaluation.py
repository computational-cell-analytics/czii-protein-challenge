import os
import csv
import argparse
from tqdm import tqdm

import numpy as np
import json

from detection.utils import metric_coords, metric_coords_per_class
from detection.data_processing.create_heatmap import parse_json_files, resolve_picks_folder


FIELDNAMES = [
    "input_name", "protein_type",
    "precision", "recall", "f1", "dev_percentage", "sMAPE", "mae",
    "matched", "label_count", "pred_count"
]


def update_csv(csv_file, new_rows):
    """Merge or update results in the evaluation CSV."""
    fieldnames = FIELDNAMES

    # Load existing data (if file exists). Rows written before the tolerance became per-class
    # are not comparable with the new ones, so move the old file aside rather than merging two
    # different metric definitions into one table.
    existing_data = {}
    stale = False
    if os.path.exists(csv_file):
        with open(csv_file, mode='r', newline='') as file:
            reader = csv.DictReader(file)
            stale = reader.fieldnames != fieldnames
            if not stale:
                for row in reader:
                    key = (row["input_name"], row["protein_type"])
                    existing_data[key] = row
        if stale:
            backup = csv_file + ".pre_per_class_radius"
            os.replace(csv_file, backup)
            print(f"{csv_file} was written with the old match distance; moved it to {backup}")

    # Update or insert new rows
    for new_row in new_rows:
        key = (new_row["input_name"], new_row["protein_type"])
        existing_data[key] = new_row  # overwrites old if exists

    # Write everything back
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(existing_data.values())


DEFAULT_RESULTS_FOLDER = os.path.join(os.path.dirname(__file__), "results")
IGNORED_LABEL_FILES = ('no_class.json', 'albumin.json', 'actin.json', 'mt.json')


def _load_labels(label_path, gt_offset=None):
    """Load the ground-truth picks, dropping the classes the detector is not trained on.

    gt_offset (dz, dy, dx) in voxels is added to every pick in memory; the stored picks are untouched.
    """
    json_files = [
        os.path.join(label_path, f)
        for f in os.listdir(label_path)
        if f.endswith('.json') and f not in IGNORED_LABEL_FILES
    ]
    coords, classes = parse_json_files(json_files)
    if gt_offset is not None and len(coords):
        coords = [tuple(c) for c in np.asarray(coords, float) + np.asarray(gt_offset, float)]
    return coords, classes


def evaluate_per_protein_type(pred_coords, label_path, model_name, input_name, gt_offset=None, results_folder=None):
    label_coords, protein_types = _load_labels(label_path, gt_offset)

    per_class_metrics = metric_coords_per_class(label_coords, protein_types, pred_coords)

    results_folder = results_folder or DEFAULT_RESULTS_FOLDER
    os.makedirs(results_folder, exist_ok=True)
    csv_file = os.path.join(results_folder, f"evaluation_{model_name}.csv")

    # Detection is class-agnostic, so a prediction has no class and per-class precision (and
    # f1 / dev_percentage / sMAPE / mae, which all compare a class count against the *total*
    # prediction count) is undefined. Those columns stay empty; see the "all" row instead.
    new_rows = []
    for protein_type, m in per_class_metrics.items():
        new_rows.append({
            "input_name": input_name,
            "protein_type": protein_type,
            "recall": m["recall"],
            "matched": m["matched"],
            "label_count": m["label_count"],
            "pred_count": len(pred_coords),
        })

    update_csv(csv_file, new_rows)
    print(f"Per-protein metrics saved to {csv_file}")


def evaluate(pred_coords, label_path, model_name, input_name, evaluate_per_protein=True, gt_offset=None,
             results_folder=None):
    label_coords, protein_types = _load_labels(label_path, gt_offset)
    with open(pred_coords, "r") as f:
        points = json.load(f)
    # Convert to NumPy array
    predictions = np.array(points)

    precision, recall, f1, dev_percentage, sMAPE, mae = metric_coords(
        label_coords, predictions, gt_classes=protein_types
    )

    results_folder = results_folder or DEFAULT_RESULTS_FOLDER
    os.makedirs(results_folder, exist_ok=True)
    csv_file = os.path.join(results_folder, f"evaluation_{model_name}.csv")

    new_row = {
        "input_name": input_name,
        "protein_type": "all",
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "dev_percentage": dev_percentage,
        "sMAPE": sMAPE,
        "mae": mae,
        "matched": round(recall * len(label_coords)),
        "label_count": len(label_coords),
        "pred_count": len(predictions),
    }
    update_csv(csv_file, [new_row])

    print(f"Metrics saved to {csv_file}")

    if evaluate_per_protein:
        evaluate_per_protein_type(predictions, label_path, model_name, input_name, gt_offset, results_folder)


def process_folder(args, file_ending):
    # Get all prediction files
    input_files = [os.path.join(args.pred_coords, name) for name in os.listdir(args.pred_coords)
                   if name.endswith(f"{file_ending}.json") and os.path.isfile(os.path.join(args.pred_coords, name))]

    # Get all label subfolders
    label_subfolders = {name: os.path.join(args.label_path, name) for name in os.listdir(args.label_path)
                        if os.path.isdir(os.path.join(args.label_path, name))}

    pbar = tqdm(input_files, desc="Evaluate files")

    for pred_coords in pbar:
        # Extract input_name from the filename
        input_name = os.path.basename(pred_coords).replace(f"{file_ending}.json", "")

        # Match with label subfolder
        label_folder = label_subfolders.get(input_name)

        if label_folder is None:
            print(f"Warning: No matching label folder found for {pred_coords}")
            continue

        evaluate(pred_coords, resolve_picks_folder(label_folder), args.model_name, input_name,
                 gt_offset=args.gt_offset, results_folder=args.output_dir)


def main():
    parser = argparse.ArgumentParser(description="Segment vesicles in EM tomograms.")
    parser.add_argument(
        "--pred_coords", "-p", required=True,
        help="The filepath to the json file of the protein location predictions."
    )
    parser.add_argument(
        "--label_path", "-l", required=True,
        help="The filepath to directory where the picked proteins are stored in a json file (gt)."
    )
    parser.add_argument(
        "--model_name", "-m", required=True, help="Name of the model."
    )

    parser.add_argument(
        "--gt_offset", nargs=3, type=float, default=None, metavar=("DZ", "DY", "DX"),
        help="Shift added to the GT picks (voxels) before matching, e.g. the public-test pick offset."
    )
    parser.add_argument(
        "--output_dir", "-o", default=None, help="Where to write the CSV. Defaults to ./results next to this script."
    )

    args = parser.parse_args()

    file_ending = "_protein_detections"

    if os.path.isfile(args.pred_coords) and args.pred_coords.endswith(".json"):
        # Extract input_name from the filename
        input_name = os.path.basename(args.pred_coords).replace(f"{file_ending}.json", "")
        evaluate(args.pred_coords, resolve_picks_folder(args.label_path), args.model_name, input_name,
                 gt_offset=args.gt_offset, results_folder=args.output_dir)
    elif os.path.isdir(args.pred_coords) and any(f.endswith(".json") for f in os.listdir(args.pred_coords)):
        process_folder(args, file_ending)
    else:
        print("Invalid input")

    print("Finished evaluating!")


if __name__ == "__main__":
    main()
