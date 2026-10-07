"""
Inspect one tomogram in napari: GT picks, each model's predictions, and an arrow from every GT pick
to its nearest prediction (so a systematic shift shows up as arrows all pointing the same way).

--gt_shift dz dy dx adds a second GT layer moved by that many voxels, e.g. the GT offset measured by
measure_pick_offset.py, to check by eye whether the corrected picks sit on the particles.

Usage:
    python detection_scripts/evaluation/view_pick_offset_napari.py \\
        -i /.../data/tomograms_VoxelSpacing10/17682 \\
        -l /.../ground_truth/structure_for_detection/tomograms_VoxelSpacing10/17682 \\
        -p v40=/.../protein_detection_czii_v40/public_test_dataset_fixed_matching/17682_protein_detections.json \\
           v44=/.../protein_detection_czii_v44/public_test_dataset/17682_protein_detections.json \\
        --gt_shift -1.4 -0.8 -0.8
"""
import argparse

import napari
import numpy as np

from detection.utils.evaluation.pick_offset import load_picks, load_predictions, match_nearest
from detection.utils.inference.gridsearch import get_volume

CLASS_COLORS = {
    "apo-ferritin": "#2a78d6",
    "beta-amylase": "#eb6834",
    "beta-galactosidase": "#1baf7a",
    "ribosome": "#eda100",
    "thyroglobulin": "#e87ba4",
    "virus-like-particle": "#008300",
}
MODEL_COLORS = ("#ffffff", "#e34948", "#00c2d1", "#c98500")
# Display size in voxels (~ particle diameter at 10 A).
POINT_SIZE = {"apo-ferritin": 12, "beta-amylase": 7, "beta-galactosidase": 14,
              "ribosome": 28, "thyroglobulin": 17, "virus-like-particle": 27}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_path", "-i", required=True, help="One tomogram folder (zarr / mrc inside).")
    ap.add_argument("--label_path", "-l", required=True, help="Its GT folder.")
    ap.add_argument("--predictions", "-p", nargs="*", default=[], metavar="NAME=JSON")
    ap.add_argument("--gt_shift", nargs=3, type=float, default=None, metavar=("DZ", "DY", "DX"))
    ap.add_argument("--max_dist", type=float, default=6.0, help="Arrow only for pairs closer than this.")
    ap.add_argument("--arrow_scale", type=float, default=3.0, help="Exaggerate arrows to make small shifts visible.")
    args = ap.parse_args()

    vol = get_volume(args.input_path)
    picks = load_picks(args.label_path)
    all_gt = np.concatenate(list(picks.values())) if picks else np.zeros((0, 3))

    v = napari.Viewer(ndisplay=2)
    lo, hi = np.percentile(vol, [1, 99])
    v.add_image(vol, name="tomogram", contrast_limits=(lo, hi))

    for cls, c in picks.items():
        if len(c):
            v.add_points(c, name=f"GT {cls}", size=POINT_SIZE.get(cls, 12), face_color="transparent",
                         border_color=CLASS_COLORS.get(cls, "white"), border_width=0.08, out_of_slice_display=True)
    if args.gt_shift is not None and len(all_gt):
        v.add_points(all_gt + np.array(args.gt_shift), name=f"GT shifted {tuple(args.gt_shift)}", size=4,
                     face_color="#eda100", border_width=0, symbol="cross", out_of_slice_display=False)

    for k, spec in enumerate(args.predictions):
        name, path = spec.split("=", 1)
        pred = load_predictions(path)
        color = MODEL_COLORS[k % len(MODEL_COLORS)]
        v.add_points(pred, name=f"pred {name}", size=4, face_color=color, border_width=0, symbol="disc",
                     out_of_slice_display=False)
        gi, pj = match_nearest(all_gt, pred, args.max_dist)
        if len(gi):
            d = pred[pj] - all_gt[gi]
            print(f"{name}: {len(gi)} pairs, mean pred - GT (dz, dy, dx) = {d.mean(0).round(2)}")
            vectors = np.stack([all_gt[gi], d * args.arrow_scale], axis=1)
            v.add_vectors(vectors, name=f"GT->{name} (x{args.arrow_scale:g})", edge_color=color,
                          edge_width=0.6, length=1.0, vector_style="arrow", out_of_slice_display=True)

    napari.run()


if __name__ == "__main__":
    main()
