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
from classification.utils import preprocess_tomo_with_labels


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
        

def run_protein_classification_with_labels(
    input_files,
    labels,
    output_path,
    model,
    idx_to_label,
    tomo_name,
    batch_size=16,
    save_full_results=True
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

    # Convert preds to labels
    all_pred_labels = [idx_to_label[p] for p in all_preds]
    all_truth_labels = [idx_to_label[t] if str(t) in idx_to_label else t for t in all_truths]

    if save_full_results:
        # Full evaluation on all subtomograms
        run_full_evaluation(all_sample_ids, all_truth_labels, all_pred_labels, all_probs, output_path, tomo_name, idx_to_label)

    return all_sample_ids, all_truth_labels, all_pred_labels, all_probs


def run_multiple_mode(args):
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

    # Optionally restrict to the test set defined in a split JSON file
    if args.split_file:
        with open(args.split_file, "r") as f:
            split = json.load(f)
        test_names = set(split["test"])
        tomogram_folders = [
            folder for folder in tomogram_folders
            if os.path.basename(folder) in test_names
        ]
        print(f"Restricting to {len(tomogram_folders)} test tomograms from {args.split_file}: "
              f"{sorted(os.path.basename(f) for f in tomogram_folders)}")
        if not tomogram_folders:
            raise ValueError(
                f"No test tomograms from {args.split_file} found in {args.input_path}."
            )

    # Pick 10% of tomograms for per-tomogram evaluation
    n_eval = max(1, int(len(tomogram_folders) * 0.1))
    eval_tomos = set(random.sample(tomogram_folders, n_eval))

    for subfolder_path in tomogram_folders:
        tomo_paths = [subfolder_path]

        print(f"Extracting subtomograms with labels from {subfolder_path}...")
        print(f"using halo of {halo}")
        subtomo_files, labels = preprocess_tomo_with_labels(
            tomo_paths,
            args.labels_root,
            max_extent,
            args.subtomo_output,
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
            save_full_results=(subfolder_path in eval_tomos)  # only full eval for 10% tomograms
        )

        global_ids.extend(sample_ids)
        global_truths.extend(truths)
        global_preds.extend(preds)
        global_probs.extend(probs)

    # run global evaluation
    run_full_evaluation(global_ids, global_truths, global_preds, global_probs, args.output_path, "ALL", idx_to_label)


def run_single_mode(args):
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
    subtomo_files, labels = preprocess_tomo_with_labels(
        tomo_paths,
        args.labels_root,
        max_extent,
        args.subtomo_output,
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
        save_full_results=True  # full evaluation in single mode
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
        "--multiple", "-mlp", action="store_true",
        help="Activate when input_path contains multiple directories for inference."
    )
    parser.add_argument(
        "--split_file", "-s", type=str, default=None,
        help="Path to a split JSON file. If given, only the tomograms listed under "
             "its 'test' key are taken from --input_path and classified."
    )

    args = parser.parse_args()
    
    if args.multiple or args.split_file:
        run_multiple_mode(args)
    else:
        run_single_mode(args)

    print("Finished classification with label evaluation!")


if __name__ == "__main__":
    main()
