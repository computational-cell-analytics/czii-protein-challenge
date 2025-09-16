import os
import csv
import argparse
from tqdm import tqdm

import numpy as np
import json

from detection.utils import metric_coords
from detection.data_processing.create_heatmap import parse_json_files

def evaluate_per_protein_type(pred_coords, label_path, model_name, input_name):
    json_files = [os.path.join(label_path, f) for f in os.listdir(label_path) if f.endswith('.json')]
    label_coords, protein_types = parse_json_files(json_files)
    
    # Organize label_coords by protein type
    label_dict = {}
    for coord, p_type in zip(label_coords, protein_types):
        if p_type not in label_dict:
            label_dict[p_type] = []
        label_dict[p_type].append(coord)
    
    results_folder = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_folder, exist_ok=True)
    csv_file = os.path.join(results_folder, f"evaluation_{model_name}.csv")
    
    # Check if the file exists to determine whether to write the header
    write_header = not os.path.exists(csv_file)
    
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow([
                "input_name", "protein_type", 
                "precision", "recall", "f1", "dev_percentage", "sMAPE", "mae", 
                "label_count", "pred_count"
            ])
        
        for protein_type, label_coords_subset in label_dict.items():
            precision, recall, f1, dev_percentage, sMAPE, mae = metric_coords(label_coords_subset, pred_coords)
            
            label_count = len(label_coords_subset)
            pred_count = len(pred_coords)
            
            writer.writerow([
                input_name, protein_type, precision, recall, f1, dev_percentage, sMAPE, mae,
                label_count, pred_count
            ])
    
    print(f"Per-protein metrics saved to {csv_file}")

def evaluate(pred_coords, label_path, model_name, input_name, evaluate_per_protein=True):
    json_files = [os.path.join(label_path, f) for f in os.listdir(label_path) if f.endswith('.json')]
    label_coords, protein_types = parse_json_files(json_files)
    with open(pred_coords, "r") as f:
        points = json.load(f)
    # Convert to NumPy array
    predictions = np.array(points)

    precision, recall, f1, dev_percentage, sMAPE, mae = metric_coords(label_coords, predictions)

    results_folder = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_folder, exist_ok=True)
    csv_file = os.path.join(results_folder, f"evaluation_{model_name}.csv")

    # Check if the file exists to determine whether to write the header
    write_header = not os.path.exists(csv_file)

    # Save the metrics to the CSV file
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow([
                "input_name", "protein_type", 
                "precision", "recall", "f1", "dev_percentage", "sMAPE", "mae", 
                "label_count", "pred_count"
            ])
        writer.writerow([
            input_name, "all", precision, recall, f1, dev_percentage, sMAPE, mae,
            len(label_coords), len(predictions)
        ])

    print(f"Metrics saved to {csv_file}")

    if evaluate_per_protein:
        evaluate_per_protein_type(predictions, label_path, model_name, input_name)

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
        label_folder = os.path.join(label_folder,"Picks")

        if label_folder:
            evaluate(pred_coords, label_folder, args.model_name, input_name)
        else:
            print(f"Warning: No matching label folder found for {pred_coords}")


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

    args = parser.parse_args()

    file_ending = "_protein_detections"#_peak_local_max"

    if os.path.isfile(args.pred_coords) and args.pred_coords.endswith(".json"):
        # Extract input_name from the filename
        input_name = os.path.basename(args.pred_coords).replace(f"{file_ending}.json", "")
        evaluate(args.pred_coords, args.label_path, args.model_name, input_name)
    elif os.path.isdir(args.pred_coords) and any(f.endswith(".json") for f in os.listdir(args.pred_coords)):
        process_folder(args, file_ending)
    else:
        print("Invalid input")

    print("Finished evaluating!")

if __name__ == "__main__":
    main()
