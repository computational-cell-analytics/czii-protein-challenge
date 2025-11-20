# visual_checks.py
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from math import ceil
from typing import List, Tuple, Dict

matplotlib.use("Agg")  # ensure headless

def _central_slices(cube: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    cube expected shape: (Z, Y, X)
    returns central XY, XZ, YZ slices as 2D arrays
    """
    z, y, x = cube.shape
    cz, cy, cx = z // 2, y // 2, x // 2
    slice_xy = cube[cz, :, :]
    slice_xz = cube[:, cy, :]
    slice_yz = cube[:, :, cx]
    return slice_xy, slice_xz, slice_yz

def save_subtomo_view(
    cube: np.ndarray,
    outpath: str,
    sample_id: str,
    pred_label: str,
    true_label: str,
    pred_prob: float = None,
    other_info: dict = None,
    cmap="gray",
    vmax=None
) -> str:
    """
    Save a 3-panel figure (XY,XZ,YZ) for one subtomogram with titles showing
    predicted label, true label and probability.
    Returns the saved filepath.
    """
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    slice_xy, slice_xz, slice_yz = _central_slices(cube)

    # auto vmax if not provided
    if vmax is None:
        vmax = max(slice_xy.max(), slice_xz.max(), slice_yz.max())

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for ax, im, title in zip(
        axs,
        [slice_xy, slice_xz, slice_yz],
        ["XY (z center)", "XZ (y center)", "YZ (x center)"]
    ):
        imshow = ax.imshow(im, cmap=cmap, origin="lower", vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    # text box with labels & probs
    prob_text = f"Prob: {pred_prob:.3f}" if pred_prob is not None else ""
    info_lines = [f"ID: {sample_id}", f"Pred: {pred_label}", f"GT: {true_label}", prob_text]
    if other_info:
        for k, v in other_info.items():
            info_lines.append(f"{k}: {v}")

    fig.suptitle("\n".join(info_lines), fontsize=10)
    cbar = fig.colorbar(imshow, ax=axs, fraction=0.02, pad=0.02)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(outpath, dpi=150)
    plt.close(fig)
    return outpath

def save_tomo_match_overview(
    tomo_volume: np.ndarray,
    pred_coords: np.ndarray,
    gt_coords: np.ndarray,
    assigned_labels: List[str],
    gt_labels_for_kept: List[str],
    outpath: str,
    projection_axis: int = 0,
    marker_size: int = 40,
    title: str = None
) -> str:
    """
    Make a simple max-projection and overlay GT vs Pred markers.
    - tomo_volume: full tomogram (Z,Y,X)
    - pred_coords: Nx3 (x,y,z) or (z,y,x)? This function expects coordinates in (x,y,z) space.
    - gt_coords: Mx3 (x,y,z)
    - assigned_labels: list len(preds) assigned label strings
    - gt_labels_for_kept: list len(gt_coords) of ground-truth labels (for visualization)
    Saves PNG at outpath and returns outpath.
    """
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    # produce max projection along projection_axis
    proj = tomo_volume.max(axis=projection_axis)

    # determine 2D coords depending on projection axis
    # projection_axis=0 -> project z -> 2D coords are (x,y) from (x,y,z)
    def to_2d(coords):
        if projection_axis == 0:
            # input coords: (x,y,z) -> 2D (x, y)
            return np.array([[c[0], c[1]] for c in coords])
        elif projection_axis == 1:
            return np.array([[c[0], c[2]] for c in coords])
        else:
            return np.array([[c[1], c[2]] for c in coords])

    pred_2d = to_2d(pred_coords) if len(pred_coords) else np.zeros((0,2))
    gt_2d = to_2d(gt_coords) if len(gt_coords) else np.zeros((0,2))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(proj, cmap="gray", origin="lower")
    # plot GT as green circles
    if len(gt_2d):
        gx, gy = gt_2d[:, 0], gt_2d[:, 1]
        ax.scatter(gx, gy, s=marker_size, facecolors='none', edgecolors='lime', label='GT')
    # plot preds colored by whether matched to a GT (assigned_labels != 'no_class') vs no_class
    if len(pred_2d):
        px, py = pred_2d[:, 0], pred_2d[:, 1]
        colors = ['red' if lbl == "no_class" else 'cyan' for lbl in assigned_labels]
        ax.scatter(px, py, s=marker_size, c=colors, marker='x', label='Pred')
    ax.legend(loc='upper right')
    if title:
        ax.set_title(title)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close(fig)
    return outpath

def make_montage_grid(
    cubes: List[np.ndarray],
    sample_ids: List[str],
    pred_labels: List[str],
    true_labels: List[str],
    probs: List[float],
    outpath: str,
    ncols: int = 5,
    thumb_size: Tuple[int,int] = (64,64)
) -> str:
    """
    Create a grid montage with one central XY slice per cube.
    """
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    n = len(cubes)
    ncols = min(ncols, max(1, n))
    nrows = ceil(n / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 2.2))
    axs = np.array(axs).reshape(-1)

    vmax = max(c.max() for c in cubes) if cubes else 1.0
    for i, ax in enumerate(axs):
        ax.axis('off')
        if i < n:
            cube = cubes[i]
            z, y, x = cube.shape
            slice_xy = cube[z//2]
            ax.imshow(slice_xy, cmap="gray", origin="lower", vmax=vmax)
            ttl = f"P:{pred_labels[i]} / G:{true_labels[i]}"
            if probs is not None:
                ttl += f"\n{probs[i]:.2f}"
            ax.set_title(ttl, fontsize=7)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close(fig)
    return outpath

def save_examples_csv(records: List[dict], out_csv: str):
    """
    records: list of dicts with keys like sample_id, h5_path, pred_label, true_label, prob, vis_path
    """
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    return out_csv

def sample_and_save_montages_by_category(
    filepaths: List[str],
    sample_ids: List[str],
    pred_labels: List[str],
    true_labels: List[str],
    probs: List[float],
    out_dir: str,
    per_category: int = 20
) -> List[str]:
    """
    Save montages for categories:
      - correct (pred==gt)
      - incorrect (pred!=gt and pred != "no_class")
      - no_class preds
    Returns list of saved montage paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    records = []
    # Helper to load central XY slice
    def _load_cube(path):
        import h5py
        with h5py.File(path, "r") as f:
            return f["raw"][:]

    # Find indices
    idx_correct = [i for i, (p, g) in enumerate(zip(pred_labels, true_labels)) if p == g]
    idx_incorrect = [i for i, (p, g) in enumerate(zip(pred_labels, true_labels)) if (p != g and p != "no_class")]
    idx_noclass = [i for i, p in enumerate(pred_labels) if p == "no_class"]

    saved = []
    for name, idxs in [("correct", idx_correct), ("incorrect", idx_incorrect), ("no_class", idx_noclass)]:
        if not idxs:
            continue
        sel = np.random.choice(idxs, size=min(per_category, len(idxs)), replace=False)
        cubes, sids, preds, trues, pr = [], [], [], [], []
        for i in sel:
            cubes.append(_load_cube(filepaths[i]))
            sids.append(sample_ids[i])
            preds.append(pred_labels[i])
            trues.append(true_labels[i])
            pr.append(probs[i] if probs is not None else None)

        outpath = os.path.join(out_dir, f"montage_{name}.png")
        save_montage = make_montage_grid(cubes, sids, preds, trues, pr, outpath)
        saved.append(save_montage)
    return saved
