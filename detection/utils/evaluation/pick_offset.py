"""Where does the density actually sit relative to a set of coordinates (GT picks or predictions)?

Coordinates are voxels of a VoxelSpacing10 tomogram, [z, y, x], same convention as parse_json_files.
"""
import json
import os

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

from detection.data_processing.create_heatmap import parse_json_files, resolve_picks_folder

IGNORED = ("no_class.json", "albumin.json", "actin.json", "mt.json")

# Half-width of the averaging box, ~1.3x the particle radius at 10 A/voxel.
BOX_HALF = {
    "apo-ferritin": 12,
    "beta-amylase": 10,
    "beta-galactosidase": 14,
    "ribosome": 20,
    "thyroglobulin": 16,
    "virus-like-particle": 20,
}
DEFAULT_BOX_HALF = 16


def load_picks(label_dir):
    """{class: (N, 3) array} for one tomogram's GT folder."""
    label_dir = resolve_picks_folder(label_dir)
    files = [os.path.join(label_dir, f) for f in os.listdir(label_dir)
             if f.endswith(".json") and f not in IGNORED]
    coords, classes = parse_json_files(files)
    coords, classes = np.asarray(coords, float).reshape(-1, 3), np.asarray(classes)
    return {c: coords[classes == c] for c in np.unique(classes)}


def load_predictions(path):
    with open(path) as f:
        return np.asarray(json.load(f), float).reshape(-1, 3)


def standardize(vol):
    vol = vol.astype(np.float32)
    return (vol - vol.mean()) / (vol.std() + 1e-8)


def sum_subtomograms(vol, centers, half):
    """Sum of boxes centred *exactly* (sub-voxel, trilinear) on each centre, and their count."""
    g = np.arange(-half, half + 1, dtype=float)
    grid = np.stack(np.meshgrid(g, g, g, indexing="ij"))
    acc, n = np.zeros(grid.shape[1:]), 0
    upper = np.array(vol.shape) - 1
    for c in np.asarray(centers, float):
        if np.any(c - half < 0) or np.any(c + half > upper):
            continue
        lo = np.floor(c - half - 1).astype(int)
        hi = np.ceil(c + half + 2).astype(int)
        sub = vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        local = grid + (c - lo)[:, None, None, None]
        acc += map_coordinates(sub, local, order=1)
        n += 1
    return acc, n


def _subvoxel_peak(cc):
    p = np.array(np.unravel_index(np.argmax(cc), cc.shape), float)
    for ax in range(3):
        i = p.astype(int)
        lo, hi = i.copy(), i.copy()
        lo[ax] -= 1
        hi[ax] += 1
        if lo[ax] < 0 or hi[ax] >= cc.shape[ax]:
            continue
        y0, y1, y2 = cc[tuple(lo)], cc[tuple(i)], cc[tuple(hi)]
        denom = y0 - 2 * y1 + y2
        if denom != 0:
            p[ax] += 0.5 * (y0 - y2) / denom
    return p


def density_centre(avg):
    """Offset (z, y, x) of the density centre from the box centre.

    Finds s with avg(s + r) ~ avg(s - r): cross-correlation with the point-reflected average peaks
    at 2s. Contrast-sign independent and exact for any (rotationally averaged) particle.
    """
    a = avg - avg.mean()
    cc = np.real(np.fft.ifftn(np.fft.fftn(a) * np.conj(np.fft.fftn(a[::-1, ::-1, ::-1]))))
    cc = np.fft.fftshift(cc)
    return (_subvoxel_peak(cc) - np.array(cc.shape) // 2) / 2


def match_nearest(gt, pred, max_dist):
    """Index pairs (gt_i, pred_j) of each GT's nearest prediction within max_dist."""
    if len(gt) == 0 or len(pred) == 0:
        return np.array([], int), np.array([], int)
    d, j = cKDTree(pred).query(gt)
    ok = d < max_dist
    return np.nonzero(ok)[0], j[ok]
