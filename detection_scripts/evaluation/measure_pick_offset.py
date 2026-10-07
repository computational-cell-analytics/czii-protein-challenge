"""
Is the shift between predictions and GT in the labels or in the model?

Averages subtomograms centred exactly on (a) the GT picks and (b) each model's predictions, and
measures where the density centre sits inside those averages. A correct coordinate set gives an
average centred at 0. Also records the raw pred - GT displacement of nearest-neighbour pairs.

  density offset at GT ~ 0, at preds != 0   -> model / pipeline bias
  density offset at GT != 0, at preds ~ 0   -> the GT is off, the model is right

Writes offsets.csv, per_tomogram.csv, displacements.csv and averages.npz to --output_dir;
plot them with plot_pick_offset.py.

Usage:
    python detection_scripts/evaluation/measure_pick_offset.py \\
        -i /.../data/tomograms_VoxelSpacing10 \\
        -l /.../ground_truth/structure_for_detection/tomograms_VoxelSpacing10 \\
        -p v40=/.../detections/protein_detection_czii_v40/public_test_dataset_fixed_matching \\
           v44=/.../detections/protein_detection_czii_v44/public_test_dataset \\
        -n 30 -o pick_offset_public_test
"""
import argparse
import csv
import os

import numpy as np

from detection.utils.evaluation.pick_offset import (
    BOX_HALF, DEFAULT_BOX_HALF, density_centre, load_picks, load_predictions, match_nearest,
    standardize, sum_subtomograms,
)
from detection.utils.inference.gridsearch import get_volume

# Pooled (all-class) box for the per-tomogram estimate.
POOLED_HALF = 16


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_path", "-i", required=True, help="Directory of tomogram folders.")
    ap.add_argument("--label_path", "-l", required=True, help="Matching ground-truth root.")
    ap.add_argument("--predictions", "-p", nargs="*", default=[], metavar="NAME=DIR",
                    help="Detection output folders (<tomogram>_protein_detections.json).")
    ap.add_argument("--n_tomograms", "-n", type=int, default=None)
    ap.add_argument("--max_dist", type=float, default=6.0,
                    help="Pred-GT pairing radius in voxels; keep below half the particle spacing.")
    ap.add_argument("--output_dir", "-o", required=True)
    args = ap.parse_args()

    models = dict(p.split("=", 1) for p in args.predictions)
    os.makedirs(args.output_dir, exist_ok=True)

    names = sorted(d for d in os.listdir(args.input_path)
                   if os.path.isdir(os.path.join(args.input_path, d))
                   and os.path.isdir(os.path.join(args.label_path, d)))[:args.n_tomograms]

    sums = {}              # (source, class) -> [sum, n]
    per_tomo, disp = [], []

    def add(key, vol, centres, half):
        a, n = sum_subtomograms(vol, centres, half)
        s = sums.setdefault(key, [0, 0])
        s[0] = s[0] + a
        s[1] += n

    for name in names:
        vol = standardize(get_volume(os.path.join(args.input_path, name)))
        picks = load_picks(os.path.join(args.label_path, name))
        all_gt = np.concatenate(list(picks.values())) if picks else np.zeros((0, 3))

        sources = {"GT": None}
        for m, d in models.items():
            f = os.path.join(d, f"{name}_protein_detections.json")
            if os.path.exists(f):
                sources[m] = load_predictions(f)

        for src, pred in sources.items():
            pooled = []
            for cls, gt in picks.items():
                if src == "GT":
                    centres = gt
                else:
                    gi, pj = match_nearest(gt, pred, args.max_dist)
                    centres = pred[pj]
                    disp += [(src, name, cls, *gt[i], *(pred[j] - gt[i])) for i, j in zip(gi, pj)]
                add((src, cls), vol, centres, BOX_HALF.get(cls, DEFAULT_BOX_HALF))
                pooled.append(centres)
            if pooled:
                a, n = sum_subtomograms(vol, np.concatenate(pooled), POOLED_HALF)
                if n >= 10:
                    per_tomo.append((src, name, n, *density_centre(a / n)))

        print(f"{name}: {len(all_gt)} GT picks, models: {[m for m in sources if m != 'GT']}")

    averages, offsets = {}, []
    for (src, cls), (a, n) in sorted(sums.items()):
        if n == 0:
            continue
        avg = a / n
        averages[f"{src}|{cls}"] = avg
        offsets.append((src, cls, n, *density_centre(avg)))
    np.savez(os.path.join(args.output_dir, "averages.npz"), **averages)

    def write(fname, header, rows):
        with open(os.path.join(args.output_dir, fname), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

    write("offsets.csv", ["source", "class", "n", "dz", "dy", "dx"], offsets)
    write("per_tomogram.csv", ["source", "tomogram", "n", "dz", "dy", "dx"], per_tomo)
    write("displacements.csv", ["model", "tomogram", "class", "gt_z", "gt_y", "gt_x", "dz", "dy", "dx"], disp)

    print("\nDensity centre relative to the coordinates (voxels; 0 = coordinates sit on the density)")
    print(f"{'source':8s} {'class':22s} {'n':>6s} {'dz':>6s} {'dy':>6s} {'dx':>6s}")
    for src, cls, n, dz, dy, dx in offsets:
        print(f"{src:8s} {cls:22s} {n:6d} {dz:+6.2f} {dy:+6.2f} {dx:+6.2f}")

    if disp:
        print("\nMean pred - GT displacement of matched pairs (voxels)")
        d = np.array([r[6:] for r in disp], float)
        m = np.array([r[0] for r in disp])
        for src in models:
            if (m == src).any():
                print(f"{src:8s} n={int((m == src).sum()):6d} mean (dz, dy, dx) = {d[m == src].mean(0).round(2)}")
    print(f"\nresults written to {args.output_dir}")


if __name__ == "__main__":
    main()
