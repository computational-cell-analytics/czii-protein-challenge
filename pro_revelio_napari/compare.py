"""Compare two density maps the way ChimeraX does.

The ChimeraX equivalents: `measure correlation` (compare_stats), `volume subtract ... minRMS true`
(match_scale + difference) and `color sample` (surface_by_value, which colours one map's isosurface
by a second map's values).

Pure numpy/scipy — no napari or Qt, so it can be used from a script or a notebook.
"""

import numpy as np

SCALE_MODES = ("min_rms", "zscore", "none")

# Shells whose mean power falls below this fraction of the strongest shell count as empty.
POWER_TOL = 1e-8


def shell_index(shape):
    """Integer Fourier-shell number for every voxel of an `fftn` output."""
    grids = np.meshgrid(*[np.fft.fftfreq(n) * n for n in shape], indexing="ij")
    return np.round(np.sqrt(sum(g**2 for g in grids))).astype(int)


def shell_power(data, shells=None):
    """Mean power per Fourier shell."""
    shells = shell_index(data.shape) if shells is None else shells
    spectrum = np.abs(np.fft.fftn(data - data.mean())) ** 2
    size = int(shells.max()) + 1
    total = np.bincount(shells.ravel(), spectrum.ravel(), minlength=size)
    count = np.bincount(shells.ravel(), minlength=size)
    return total / np.maximum(count, 1)


def power_cutoff(data, tol=POWER_TOL):
    """Highest Fourier shell that still carries signal — this is what detects a low-pass filter."""
    power = shell_power(data)[1:]
    if power.size == 0 or power.max() <= 0:
        return 1
    above = np.nonzero(power > tol * power.max())[0]
    return int(above[-1]) + 1 if above.size else 1


def lowpass(data, cutoff, edge=2):
    """Raised-cosine low-pass at `cutoff` shells."""
    shells = shell_index(data.shape).astype(np.float32)
    ramp = np.clip((cutoff - shells) / max(edge, 1e-6), 0.0, 1.0)
    mask = 0.5 - 0.5 * np.cos(np.pi * ramp)
    return np.real(np.fft.ifftn(np.fft.fftn(data) * mask)).astype(np.float32)


def match_bandlimit(a, b, tol=POWER_TOL):
    """Filter the sharper map down to the blunter one's resolution.

    Skipping this makes the difference map show the filter rather than the structure: frequencies
    that only one map contains dominate everything else.
    """
    cutoffs = (power_cutoff(a, tol), power_cutoff(b, tol))
    cutoff = min(cutoffs)
    if cutoffs[0] != cutoff:
        a = lowpass(a, cutoff)
    if cutoffs[1] != cutoff:
        b = lowpass(b, cutoff)
    return a, b, cutoffs, cutoff


def match_scale(a, b, mode="min_rms"):
    """Put the two maps on a common amplitude scale. Returns the maps and the factor applied to b.

    `min_rms` is ChimeraX's `volume subtract ... minRMS true`: scale b so ||a - b|| is minimal.
    """
    if mode == "none":
        return a, b, 1.0

    a = a - a.mean()
    b = b - b.mean()
    if mode == "zscore":
        factor = 1.0 / (float(b.std()) or 1.0)
        return a / (float(a.std()) or 1.0), b * factor, factor

    denom = float((b * b).sum())
    factor = float((a * b).sum()) / denom if denom else 1.0
    return a, b * factor, factor


def align_shift(a, b):
    """Sub-voxel shift that moves b onto a, from the cross-correlation peak."""
    cc = np.real(np.fft.ifftn(np.fft.fftn(a - a.mean()) * np.conj(np.fft.fftn(b - b.mean()))))
    peak = np.unravel_index(int(np.argmax(cc)), cc.shape)

    shift = []
    for axis, position in enumerate(peak):
        size = cc.shape[axis]
        before = cc[_neighbour(peak, axis, -1, cc.shape)]
        after = cc[_neighbour(peak, axis, +1, cc.shape)]
        curvature = before - 2 * cc[peak] + after
        delta = 0.5 * (before - after) / curvature if curvature else 0.0
        wrapped = position - size if position > size // 2 else position
        shift.append(float(wrapped + delta))
    return tuple(shift)


def apply_shift(data, shift):
    """Shift in Fourier space — exact, and consistent with how `align_shift` measured it."""
    from scipy.ndimage import fourier_shift

    return np.real(np.fft.ifftn(fourier_shift(np.fft.fftn(data), shift))).astype(np.float32)


def prepare(a, b, scale="min_rms", match_band=True, align=False):
    """Run the preprocessing ChimeraX would have you do by hand before subtracting."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"maps must have the same shape, got {a.shape} and {b.shape}")

    info = {"cutoffs": None, "cutoff": None, "factor": 1.0, "shift": None}
    if match_band:
        a, b, info["cutoffs"], info["cutoff"] = match_bandlimit(a, b)
    if align:
        info["shift"] = align_shift(a, b)
        b = apply_shift(b, info["shift"])
    a, b, info["factor"] = match_scale(a, b, scale)
    return a, b, info


def compare_stats(a, b, level_sigma=2.0):
    """ChimeraX `measure correlation`, plus a threshold-based overlap."""
    threshold = float(a.mean() + level_sigma * a.std())
    mask_a, mask_b = a > threshold, b > threshold
    total = int(mask_a.sum()) + int(mask_b.sum())
    return {
        "cc": float(np.corrcoef(a.ravel(), b.ravel())[0, 1]),
        "rms": float(np.sqrt(np.mean((a - b) ** 2))),
        "dice": 2 * float((mask_a & mask_b).sum()) / total if total else 0.0,
        "threshold": threshold,
        "voxels_a": int(mask_a.sum()),
        "voxels_b": int(mask_b.sum()),
    }


def surface_by_value(data, values, level):
    """ChimeraX `color sample`: isosurface of `data`, one sampled value per vertex.

    Returns (vertices, faces, values) ready for `viewer.add_surface`.
    """
    from scipy.ndimage import map_coordinates
    from skimage.measure import marching_cubes

    vertices, faces, _, _ = marching_cubes(np.asarray(data), level=level)
    sampled = map_coordinates(values, vertices.T, order=1, mode="nearest")
    return vertices, faces, sampled.astype(np.float32)


def agreement_labels(a, b, level_sigma=2.0):
    """Thresholded overlap as a label volume: 1 = both, 2 = only A, 3 = only B."""
    threshold = a.mean() + level_sigma * a.std()
    mask_a, mask_b = a > threshold, b > threshold
    labels = np.zeros(a.shape, dtype=np.uint8)
    labels[mask_a & mask_b] = 1
    labels[mask_a & ~mask_b] = 2
    labels[~mask_a & mask_b] = 3
    return labels


def resolution(shell, size, apix):
    """Fourier shell number to Ångström, for reporting."""
    return size * apix / shell if shell else float("inf")


def _neighbour(index, axis, step, shape):
    out = list(index)
    out[axis] = (out[axis] + step) % shape[axis]
    return tuple(out)


def main(argv=None):
    """Open two maps side by side with the comparison panel: python -m pro_revelio_napari.compare A B"""
    import argparse

    parser = argparse.ArgumentParser(description="Compare two density maps in napari.")
    parser.add_argument("map_a")
    parser.add_argument("map_b")
    parser.add_argument("--scale", default="min_rms", choices=SCALE_MODES)
    parser.add_argument("--no-match-band", action="store_true", help="do not equalise the resolutions")
    parser.add_argument("--align", action="store_true", help="shift map B onto map A first")
    parser.add_argument("--level", type=float, default=2.0, help="isosurface level in sigma")
    args = parser.parse_args(argv)

    import napari

    viewer = napari.Viewer()
    layer_a = viewer.open(args.map_a, plugin="pro-revelio")[0]
    layer_b = viewer.open(args.map_b, plugin="pro-revelio")[0]
    _, widget = viewer.window.add_plugin_dock_widget("pro-revelio", "Compare Maps")

    widget.configure(
        map_a=layer_a.name,
        map_b=layer_b.name,
        scale=args.scale,
        match_band=not args.no_match_band,
        align=args.align,
        level=args.level,
    )
    widget.run()
    napari.run()


if __name__ == "__main__":
    main()
