"""
How high does the model's heatmap actually rise at a real particle, per class?

The training target puts every class's Gaussian peak at 3.989, but the predictions do not get
close - which is why the threshold gridsearch keeps pinning at the bottom of its sweep range.
This answers two questions that decide what to do about it:

  1. Is the shortfall uniform across classes? If so it is only a scaling issue and lowering
     GRIDSEARCH_THRESH_RANGE is enough.
  2. Or is it class-dependent? If beta-amylase peaks sit near the noise floor while ribosome
     peaks are high, no single threshold can serve both and the fix has to be in training.

For every ground-truth pick it takes the maximum predicted heatmap value in a small window
around the pick - i.e. the response the model gives that particle, whether or not it ever
became a detection - and reports the distribution per class against the background level.

Usage:
    python detection_scripts/evaluation/measure_peak_response.py \\
        -m /.../models/checkpoints/protein_detection_czii_v43 \\
        -i /.../data/tomograms_VoxelSpacing10 \\
        -l /.../ground_truth/structure_for_detection/tomograms_VoxelSpacing10 \\
        -n 3 -o peak_response_v43.csv
"""
import argparse
import os

import numpy as np

from detection.data_processing.create_heatmap import parse_json_files, resolve_picks_folder
from detection.utils.prediction.prediction import get_prediction_torch_em, load_detection_model
from detection.utils.training.tiling_helper import parse_tiling
from detection.utils.inference.gridsearch import get_volume
from detection.config import GRIDSEARCH_THRESH_RANGE

IGNORED = ("no_class.json", "albumin.json", "actin.json", "mt.json")
# Half-width of the window searched around each pick. Generous enough to absorb the ~1.5 voxel
# pick-vs-density offset in the public test set without reaching a neighbouring particle.
WINDOW = 4


def picks_for(label_dir):
    files = [os.path.join(label_dir, f) for f in os.listdir(label_dir)
             if f.endswith(".json") and f not in IGNORED]
    coords, classes = parse_json_files(files)
    return np.array(coords), np.array(classes)


def response_at(heatmap, coords, window=WINDOW):
    """Max predicted heatmap value in a +/-window box around each coordinate."""
    out = []
    for c in np.round(coords).astype(int):
        z, y, x = c
        zs = slice(max(0, z - window), min(heatmap.shape[0], z + window + 1))
        ys = slice(max(0, y - window), min(heatmap.shape[1], y + window + 1))
        xs = slice(max(0, x - window), min(heatmap.shape[2], x + window + 1))
        sub = heatmap[zs, ys, xs]
        out.append(float(sub.max()) if sub.size else np.nan)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_path", "-m", required=True)
    ap.add_argument("--input_path", "-i", required=True, help="Directory of tomogram folders.")
    ap.add_argument("--label_path", "-l", required=True, help="Matching ground-truth root.")
    ap.add_argument("--n_tomograms", "-n", type=int, default=3)
    ap.add_argument("--split_json", "-s", default=None,
                    help="A split-*.json; measure on its 'val' list instead of the first n "
                         "tomograms. Use this on the training dataset, or the response is read "
                         "off tomograms the model was fitted to.")
    ap.add_argument("--split_key", default="val", choices=["train", "val", "test"])
    ap.add_argument("--output", "-o", default=None, help="Optional CSV of the per-particle responses.")
    args = ap.parse_args()

    model = load_detection_model(args.model_path)
    tiling = parse_tiling(tile_shape=None, halo=None)

    if args.split_json:
        import json
        with open(args.split_json) as f:
            names = json.load(f)[args.split_key][:args.n_tomograms]
        print(f"measuring on the '{args.split_key}' split: {names}")
    else:
        names = sorted(d for d in os.listdir(args.input_path)
                       if os.path.isdir(os.path.join(args.input_path, d)))[:args.n_tomograms]

    rows = []          # (tomogram, class, response)
    background = []    # sampled background level, for reference
    for name in names:
        label_dir = resolve_picks_folder(os.path.join(args.label_path, name))
        if not os.path.isdir(label_dir):
            print(f"no labels for {name}, skipping")
            continue

        volume = get_volume(os.path.join(args.input_path, name))
        pred = get_prediction_torch_em(input_volume=volume, tiling=tiling, model=model, verbose=True)
        heatmap = pred[0]

        coords, classes = picks_for(label_dir)
        resp = response_at(heatmap, coords)
        rows += [(name, c, r) for c, r in zip(classes, resp)]

        rng = np.random.default_rng(0)
        bg = np.stack([rng.integers(WINDOW, s - WINDOW, 2000) for s in heatmap.shape], axis=1)
        background.append(response_at(heatmap, bg))
        print(f"{name}: {len(coords)} picks, heatmap max {heatmap.max():.3f}, mean {heatmap.mean():.4f}")

    if not rows:
        print("nothing measured")
        return

    background = np.concatenate(background)
    classes = np.array([r[1] for r in rows])
    resp = np.array([r[2] for r in rows], dtype=float)

    print(f"\ntarget peak is 3.989 for every class; gridsearch sweeps {GRIDSEARCH_THRESH_RANGE}")
    print(f"background response: median {np.median(background):.3f}, p99 {np.percentile(background, 99):.3f}\n")
    print(f"{'class':24s} {'n':>5s} {'p10':>7s} {'p50':>7s} {'p90':>7s} {'max':>7s} {'>bg p99':>8s} {'>1.0':>7s}")
    for c in sorted(set(classes)):
        v = resp[classes == c]
        v = v[~np.isnan(v)]
        if not len(v):
            continue
        print(f"{c:24s} {len(v):5d} {np.percentile(v,10):7.3f} {np.percentile(v,50):7.3f} "
              f"{np.percentile(v,90):7.3f} {v.max():7.3f} "
              f"{100*(v > np.percentile(background,99)).mean():7.1f}% {100*(v > 1.0).mean():6.1f}%")

    if args.output:
        import csv
        with open(args.output, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["tomogram", "class", "peak_response"])
            w.writerows(rows)
        print(f"\nper-particle responses written to {args.output}")


if __name__ == "__main__":
    main()
