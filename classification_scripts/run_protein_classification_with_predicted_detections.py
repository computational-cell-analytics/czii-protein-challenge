import argparse
import os
import h5py
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import random
import warnings

from classification.utils import protein_classification, get_model
from classification.utils import preprocess_tomo_with_predictions

from classification.inference.visual_checks import save_subtomo_view


def summarize_distance_bins(distances):
    """
    distances: 1D iterable of floats
    
    Returns a dict with counts for fixed distance ranges.
    """
    bins = {
        "0-5": 0,
        "5-10": 0,
        "10-15": 0,
        "15-20": 0,
        "20-25": 0,
        "25-30": 0,
        "30-35": 0,
        "35-40": 0,
        "40-45": 0,
        "45-50": 0,
        ">50": 0
    }

    for d in distances:
        if d < 5:
            bins["0-5"] += 1
        elif d < 10:
            bins["5-10"] += 1
        elif d < 15:
            bins["10-15"] += 1
        elif d < 20:
            bins["15-20"] += 1
        elif d < 25:
            bins["20-25"] += 1
        elif d < 30:
            bins["25-30"] += 1
        elif d < 35:
            bins["30-35"] += 1
        elif d < 40:
            bins["35-40"] += 1
        elif d < 45:
            bins["40-45"] += 1
        elif d < 50:
            bins["45-50"] += 1
        else:
            bins[">50"] += 1

    return bins


def run_full_evaluation(sample_ids, truth_labels, pred_labels, probs, output_path, name, idx_to_label, num_unmatched, unmatched_info=None):
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
    # Save results as list (CSV) 
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
    
    # Write summary including unmatched predictions (= no_class)
    summary_path = os.path.join(output_path, f"summary_{name}.csv")
    write_header = not os.path.exists(summary_path)
    with open(summary_path, "a") as f:
        if write_header:
            f.write("tomo_name,total_preds,matched_preds,unmatched_preds\n")
        f.write(f"{name},{len(sample_ids)+num_unmatched},{len(sample_ids)},{num_unmatched}\n")

    if unmatched_info:
        # Save unmatched_info per tomogram 
        # Collect global distances
        global_distances = []

        per_tomo_summary = {}
        for tomo_name, info in unmatched_info.items():
            dists = info["distances"]
            summary = summarize_distance_bins(dists)
            per_tomo_summary[tomo_name] = summary
            global_distances.extend(dists)

        # Global summary
        global_summary = summarize_distance_bins(global_distances)

        #save distance summary
        json_path = os.path.join(output_path, f"distance_summary_{name}.json")
        with open(json_path, "w") as f:
            json.dump(
                {
                    "per_tomogram": per_tomo_summary,
                    "global": global_summary
                },
                f,
                indent=4
            )


def run_protein_classification_with_labels(
    input_files,
    labels,
    output_path,
    model,
    idx_to_label,
    tomo_name,
    batch_size=16,
    save_full_results=True,
    num_unmatched=None,
    unmatched_info=None
):
    os.makedirs(output_path, exist_ok=True)

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
        probs, preds = protein_classification(cubes, model, EfficientNet=False)

        all_sample_ids.extend(sample_ids)
        all_preds.extend(preds)
        all_probs.extend(probs.tolist())
        all_truths.extend(truths_batch)

    #Convert int preds to labels 
    all_pred_labels = [idx_to_label[p] for p in all_preds]
    all_truth_labels = [idx_to_label[t] if str(t) in idx_to_label else t for t in all_truths]

    if save_full_results:
        # Full evaluation on all subtomograms
        run_full_evaluation(all_sample_ids, all_truth_labels, all_pred_labels, all_probs, output_path, tomo_name, idx_to_label, num_unmatched, unmatched_info)

    # set up visualization dir
    vis_dir = os.path.join(output_path, "visual_checks")
    os.makedirs(vis_dir, exist_ok=True)

    # per-sample visualization and csv results (do for the current batch)
    batch_records = []
    for sid, cube, pred_int, prob_vec, true_lbl in zip(sample_ids, cubes, preds, probs, truths_batch):

        # Decide top probability and label (if probs is vector)
        if hasattr(prob_vec, "__len__") and len(prob_vec) > 1:
            top_prob = float(np.max(prob_vec))
        else:
            top_prob = float(prob_vec) if not hasattr(prob_vec, "__len__") else float(prob_vec[0])

        # translate pred_int to label string using idx_to_label
        pred_label_str = idx_to_label[pred_int] if 'idx_to_label' in globals() and str(pred_int) in idx_to_label else str(pred_int)
        true_label_str = idx_to_label[true_lbl] if 'idx_to_label' in globals() and str(true_lbl) in idx_to_label else str(true_lbl)

        # sample-specific viz path
        viz_path = os.path.join(vis_dir, f"{sid}_viz.png")
        save_subtomo_view(cube, viz_path, sid, pred_label_str, true_label_str, pred_prob=top_prob)

        batch_records.append({
            "sample_id": sid,
            "h5_path": sid,
            "pred_label": pred_label_str,
            "true_label": true_label_str,
            "prob": top_prob,
            "visualization": viz_path
        })

    # write/append CSV for this batch
    csv_path = os.path.join(vis_dir, "examples_index.csv")
    # load existing if any and append
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df_new = pd.DataFrame(batch_records)
        pd.concat([df_existing, df_new], ignore_index=True).to_csv(csv_path, index=False)
    else:
        pd.DataFrame(batch_records).to_csv(csv_path, index=False)

    return all_sample_ids, all_truth_labels, all_pred_labels, all_probs


def run_multi_tomogram_mode(args):

    model_path = args.model_path

    if model_path.endswith("best.pt"):
        model_path = os.path.split(model_path)[0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, checkpoint = get_model(
            model_path=model_path,
            device="cpu",
            EfficientNet=False
        )
    
    init_data = checkpoint['init']

    # access custom parameters
    patch_shape = init_data.get('patch_shape')
    idx_to_label = init_data.get('idx_to_label')
    idx_to_label = {k: str(v) for k, v in idx_to_label.items()}

    print(f"Patch shape: {patch_shape}")
    print(f"Index to label mapping: {idx_to_label}")

    max_extent = patch_shape[0]
    halo = 0

    '''#backup
    if idx_to_label is None:
        print("needed idx_to_lable backup")
        # Replace 'models/checkpoints/' with 'training' and build path to idx_to_label.json; not ideal, but otherwise I'd have to add another argument parser. maybe thats better?
        idx_file_path = os.path.join(
            args.model_path.replace("/models/checkpoints/", "/training/"),
            "idx_to_label.json"
        )

        # Load JSON
        with open(idx_file_path, "r") as f:
            idx_to_label = json.load(f)
    
    if max_extent is None:
        print("no max_extent")
        max_extent=33
        halo=None'''
            
    global_ids, global_truths, global_preds, global_probs = [], [], [], []

    # Collect all tomogram subfolders
    tomogram_folders = [
        os.path.join(args.input_path, sf)
        for sf in os.listdir(args.input_path)
        if os.path.isdir(os.path.join(args.input_path, sf))
    ]

    # Pick 10% of tomograms for per-tomogram evaluation
    n_eval = max(1, int(len(tomogram_folders) * 0.1))
    eval_tomos = set(random.sample(tomogram_folders, n_eval))

    global_num_unmatched = 0
    global_unmatched_info = {}

    for subfolder_path in tomogram_folders:
        tomo_paths = [subfolder_path]
        tomoID = os.path.basename(subfolder_path)
        base_folder = os.path.basename(os.path.dirname(subfolder_path))

        print(f"Extracting subtomograms with labels from {subfolder_path}...")
        subtomo_files, labels, num_unmatched, unmatched_info = preprocess_tomo_with_predictions(
            tomo_paths,
            pred_path=os.path.join(args.pred_coords, f"{tomoID}_protein_detections.json"),
            label_path=os.path.join(args.labels_root, base_folder, tomoID, "Picks"),
            max_extent=max_extent,
            subtomo_output=args.subtomo_output,
            halo=halo
        )

        tomo_name = os.path.basename(subfolder_path)

        print(f"Classifying {len(subtomo_files)} subtomograms from {subfolder_path}...")
        sample_ids, truths, preds, probs = run_protein_classification_with_labels(
            subtomo_files,
            labels,
            args.output_path,
            model,
            idx_to_label,
            tomo_name=tomo_name,
            batch_size=args.batch_size,
            save_full_results=(subfolder_path in eval_tomos),  # only full eval for 10% tomograms
            num_unmatched=num_unmatched,
            unmatched_info=None
        )

        global_ids.extend(sample_ids)
        global_truths.extend(truths)
        global_preds.extend(preds)
        global_probs.extend(probs)
        global_num_unmatched += num_unmatched
        global_unmatched_info.update(unmatched_info)

    # Global evaluation
    run_full_evaluation(global_ids, global_truths, global_preds, global_probs, args.output_path, "ALL", idx_to_label, global_num_unmatched, global_unmatched_info)


def run_single_tomogram_mode(args):
    model_path = args.model_path

    if model_path.endswith("best.pt"):
        model_path = os.path.split(model_path)[0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, checkpoint = get_model(
            model_path=model_path,
            device="cpu",
            EfficientNet=False
        )
    init_data = checkpoint['init']

    # access custom parameters
    patch_shape = init_data.get('patch_shape')
    idx_to_label = init_data.get('idx_to_label')
    idx_to_label = {k: str(v) for k, v in idx_to_label.items()}

    print(f"Patch shape: {patch_shape}")
    print(f"Index to label mapping: {idx_to_label}")

    max_extent = patch_shape[0]
    halo = 0

    '''#backup
    if idx_to_label is None:
        print("needed idx_to_lable backup")
        # Replace 'models/checkpoints/' with 'training' and build path to idx_to_label.json; not ideal, but otherwise I'd have to add another argument parser. maybe thats better?
        idx_file_path = os.path.join(
            args.model_path.replace("/models/checkpoints/", "/training/"),
            "idx_to_label.json"
        )

        # Load JSON
        with open(idx_file_path, "r") as f:
            idx_to_label = json.load(f)
    
    if max_extent is None:
        print("no max_extent")
        max_extent=33
        halo=None'''

    tomo_paths = [args.input_path]

    print("Extracting subtomograms with labels...")
    subtomo_files, labels, num_unmatched, unmatched_info = preprocess_tomo_with_predictions(
        tomo_paths,
        pred_path=args.pred_coords,
        label_path=os.path.join(args.labels_root, "Picks"),
        max_extent=max_extent,
        subtomo_output=args.subtomo_output,
        halo=halo
    )

    tomo_name = os.path.basename(args.input_path)

    print(f"Classifying {len(subtomo_files)} subtomograms...")
    run_protein_classification_with_labels(
        subtomo_files,
        labels,
        args.output_path,
        model,
        idx_to_label,
        tomo_name=tomo_name,
        batch_size=args.batch_size,
        save_full_results=True,  # full evaluation in single mode,
        num_unmatched=num_unmatched,
        unmatched_info=unmatched_info
    )


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
        run_multi_tomogram_mode(args)
    else:
        run_single_tomogram_mode(args)

    print("Finished classification with label evaluation!")


if __name__ == "__main__":
    main()
