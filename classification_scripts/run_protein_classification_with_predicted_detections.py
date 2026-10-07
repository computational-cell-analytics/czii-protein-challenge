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


# Distance to the nearest ground-truth pick, in voxels. Resolved finely below ~8 voxels
# because that is where the CZII tolerances sit (3.0 apo-ferritin to 7.5 ribosome).
DISTANCE_BIN_EDGES = [1, 2, 3, 4, 5, 6, 7.5, 10, 15, 20, 30, 50]


def summarize_distance_bins(distances):
    """
    distances: 1D iterable of floats (voxels)

    Returns a dict with counts for fixed distance ranges.
    """
    labels = ["0-1"] + [
        f"{DISTANCE_BIN_EDGES[i - 1]:g}-{edge:g}" for i, edge in enumerate(DISTANCE_BIN_EDGES[1:], start=1)
    ] + [f">{DISTANCE_BIN_EDGES[-1]:g}"]
    bins = {label: 0 for label in labels}

    for d in distances:
        for i, edge in enumerate(DISTANCE_BIN_EDGES):
            if d < edge:
                bins[labels[i]] += 1
                break
        else:
            bins[labels[-1]] += 1

    return bins


SUMMARY_COLUMNS = ["total_preds", "kept_crops", "matched_preds", "unmatched_preds", "dropped_at_border"]


def merge_counts(*counts):
    """Sum the per-tomogram prediction counts into a global one."""
    return {col: sum(c.get(col, 0) for c in counts) for col in SUMMARY_COLUMNS}


def run_full_evaluation(sample_ids, truth_labels, pred_labels, probs, output_path, name, idx_to_label, counts, unmatched_info=None):
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

    # Save confusion matrix (row-normalized, i.e. per-class recall). Classes with no ground
    # truth in this run have an empty row and stay at 0 instead of turning into NaN.
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm.astype("float"), row_sums, out=np.zeros(cm.shape, dtype=float), where=row_sums != 0
    )
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", xticklabels=idx_to_label.values(), yticklabels=idx_to_label.values(), cmap="Blues", vmin=0, vmax=1)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Row-normalized Confusion Matrix (= recall) - {name}")
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
    
    # Write summary. kept_crops = matched + unmatched, and total_preds - kept_crops are the
    # predictions whose box left the volume and never reached the classifier.
    summary_path = os.path.join(output_path, f"summary_{name}.csv")
    with open(summary_path, "w") as f:
        f.write("tomo_name," + ",".join(SUMMARY_COLUMNS) + "\n")
        f.write(f"{name}," + ",".join(str(counts[col]) for col in SUMMARY_COLUMNS) + "\n")

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


VIS_SEED = 0


def reset_visual_checks(output_path):
    """Drop a stale example index so a re-run does not append to the previous run's rows."""
    csv_path = os.path.join(output_path, "visual_checks", "examples_index.csv")
    if os.path.exists(csv_path):
        os.remove(csv_path)


def select_visualization_indices(n_crops, n_samples, tomo_name):
    """
    Pick which crops of a tomogram get a rendered view.

    Seeded on the tomogram name, so the sample is reproducible and independent of the order
    the tomograms happen to be processed in. n_samples < 0 means all, 0 means none.
    """
    if n_samples is not None and n_samples == 0:
        return set()
    if n_samples is None or n_samples < 0 or n_samples >= n_crops:
        return set(range(n_crops))
    return set(random.Random(f"{VIS_SEED}:{tomo_name}").sample(range(n_crops), n_samples))


def run_protein_classification_with_labels(
    input_files,
    labels,
    output_path,
    model,
    idx_to_label,
    tomo_name,
    batch_size=16,
    save_full_results=True,
    counts=None,
    unmatched_info=None,
    vis_samples=16
):
    os.makedirs(output_path, exist_ok=True)

    all_sample_ids, all_preds, all_probs, all_truths = [], [], [], []

    # Decide upfront which crops to render, so the cubes can be held on to as they stream past
    # instead of being re-read afterwards.
    vis_indices = select_visualization_indices(len(input_files), vis_samples, tomo_name)
    vis_cubes = {}

    # Batch processing
    for i in range(0, len(input_files), batch_size):
        batch_paths = input_files[i:i + batch_size]
        cubes, sample_ids, truths_batch = [], [], []

        for j, (path, label) in enumerate(zip(batch_paths, labels[i:i + batch_size])):
            with h5py.File(path, "r") as f:
                cubes.append(f["raw"][:])
            if i + j in vis_indices:
                vis_cubes[i + j] = cubes[-1]
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
        run_full_evaluation(all_sample_ids, all_truth_labels, all_pred_labels, all_probs, output_path, tomo_name, idx_to_label, counts, unmatched_info)

    # Per-sample visualization, over the sample drawn from *all* crops of this tomogram.
    # Labels come from the already-translated lists, which are aligned with input_files.
    if vis_cubes:
        vis_dir = os.path.join(output_path, "visual_checks")
        os.makedirs(vis_dir, exist_ok=True)

        records = []
        for idx in sorted(vis_cubes):
            sid = all_sample_ids[idx]
            top_prob = float(np.max(all_probs[idx]))
            viz_path = os.path.join(vis_dir, f"{sid}_viz.png")
            save_subtomo_view(
                vis_cubes[idx], viz_path, sid,
                all_pred_labels[idx], all_truth_labels[idx], pred_prob=top_prob,
            )

            records.append({
                "sample_id": sid,
                "h5_path": sid,
                "pred_label": all_pred_labels[idx],
                "true_label": all_truth_labels[idx],
                "prob": top_prob,
                "visualization": viz_path
            })

        # One block of rows per tomogram; reset_visual_checks clears the file between runs.
        csv_path = os.path.join(vis_dir, "examples_index.csv")
        pd.DataFrame(records).to_csv(
            csv_path, mode="a", header=not os.path.exists(csv_path), index=False
        )

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

    reset_visual_checks(args.output_path)
    global_counts = []
    global_unmatched_info = {}

    for subfolder_path in tomogram_folders:
        tomo_paths = [subfolder_path]
        tomoID = os.path.basename(subfolder_path)
        base_folder = os.path.basename(os.path.dirname(subfolder_path))

        print(f"Extracting subtomograms with labels from {subfolder_path}...")
        subtomo_files, labels, counts, unmatched_info = preprocess_tomo_with_predictions(
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
            counts=counts,
            unmatched_info=None,
            vis_samples=args.vis_samples
        )

        global_ids.extend(sample_ids)
        global_truths.extend(truths)
        global_preds.extend(preds)
        global_probs.extend(probs)
        global_counts.append(counts)
        global_unmatched_info.update(unmatched_info)

    # Global evaluation
    run_full_evaluation(global_ids, global_truths, global_preds, global_probs, args.output_path, "ALL", idx_to_label, merge_counts(*global_counts), global_unmatched_info)


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

    reset_visual_checks(args.output_path)

    tomo_paths = [args.input_path]

    print("Extracting subtomograms with labels...")
    subtomo_files, labels, counts, unmatched_info = preprocess_tomo_with_predictions(
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
        counts=counts,
        unmatched_info=unmatched_info,
        vis_samples=args.vis_samples
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
    parser.add_argument(
        "--vis_samples", type=int, default=16,
        help="Rendered example crops per tomogram, drawn at random with a fixed seed. "
             "0 disables them, a negative value renders every crop."
    )

    args = parser.parse_args()
    
    if os.path.isdir(args.pred_coords):
        run_multi_tomogram_mode(args)
    else:
        run_single_tomogram_mode(args)

    print("Finished classification with label evaluation!")


if __name__ == "__main__":
    main()
