"""
Figures for the output of measure_pick_offset.py.

  averages_<class>.png        orthoslices of the average at GT / at each model's predictions;
                              crosshair = where the coordinates point, dot = measured density centre
  displacement_hist.png       pred - GT per axis and model; dashed = where a model that hits the
                              density exactly would land, given the GT's own offset
  displacement_vs_position.png  pred - GT against position: flat = constant shift, sloped = scale error
  per_tomogram.png            density offset per tomogram: one outlier tomogram vs. a global shift

Usage:
    python detection_scripts/evaluation/plot_pick_offset.py -r pick_offset_public_test
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

AXES = ("z", "y", "x")
AXIS_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
SOURCE_COLORS = ("#6b6b66", "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
INK, MUTED, GRID = "#1f1f1e", "#6b6b66", "#e4e3dd"

plt.rcParams.update({
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "font.size": 9, "axes.titlesize": 9,
})


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def source_order(rows, key):
    seen = []
    for r in rows:
        if r[key] not in seen:
            seen.append(r[key])
    return (["GT"] if "GT" in seen else []) + [s for s in seen if s != "GT"]


def plot_averages(res, out, offsets):
    data = np.load(os.path.join(res, "averages.npz"))
    off = {(r["source"], r["class"]): np.array([float(r[a]) for a in ("dz", "dy", "dx")]) for r in offsets}
    classes = sorted({k.split("|")[1] for k in data.files})
    sources = source_order(offsets, "source")
    for cls in classes:
        srcs = [s for s in sources if f"{s}|{cls}" in data.files]
        fig, axs = plt.subplots(len(srcs), 4, figsize=(11, 2.7 * len(srcs)), squeeze=False)
        for row, src in zip(axs, srcs):
            avg = data[f"{src}|{cls}"]
            c = avg.shape[0] // 2
            o = off[(src, cls)]
            # 3-slice slabs for SNR; (array, row-axis, col-axis) per view
            views = [("XY", avg[c - 1:c + 2].mean(0), 1, 2),
                     ("XZ", avg[:, c - 1:c + 2].mean(1), 0, 2),
                     ("YZ", avg[:, :, c - 1:c + 2].mean(2), 0, 1)]
            lim = np.percentile(np.abs(avg - np.median(avg)), 99.5)
            for ax, (title, img, ra, ca) in zip(row, views):
                ax.imshow(img - np.median(avg), cmap="gray", vmin=-lim, vmax=lim, origin="lower")
                ax.axhline(c, color="#eda100", lw=0.8)
                ax.axvline(c, color="#eda100", lw=0.8)
                ax.plot(c + o[ca], c + o[ra], "o", ms=6, mfc="#e34948", mec="white", mew=1.5)
                ax.set_title(f"{src} - {title}", loc="left")
                ax.set_xlabel(AXES[ca])
                ax.set_ylabel(AXES[ra])
                ax.grid(False)
            # density profile along each axis, through a central slab
            ax = row[3]
            sign = -1 if avg[c - 2:c + 3, c - 2:c + 3, c - 2:c + 3].mean() < avg.mean() else 1
            core = sign * (avg - np.median(avg))
            r = np.arange(avg.shape[0]) - c
            for k, (name, col) in enumerate(zip(AXES, AXIS_COLORS)):
                sl = [slice(c - 2, c + 3)] * 3
                sl[k] = slice(None)
                ax.plot(r, core[tuple(sl)].mean(tuple(i for i in range(3) if i != k)), color=col, lw=2,
                        label=f"{name}: {o[k]:+.2f}")
            ax.axvline(0, color=MUTED, lw=0.8, ls="--")
            ax.set_xlabel("voxels from coordinate")
            ax.set_title("profile (density centre offset)", loc="left")
            ax.legend(frameon=False, fontsize=8)
        fig.suptitle(f"{cls}: average at coordinates (crosshair) vs density centre (dot)", x=0.01, ha="left",
                     color=INK)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"averages_{cls}.png"), dpi=130)
        plt.close(fig)


def gt_offset(offsets):
    """Count-weighted density offset at the GT picks, all classes."""
    rows = [r for r in offsets if r["source"] == "GT"]
    if not rows:
        return None
    w = np.array([float(r["n"]) for r in rows])
    v = np.array([[float(r[a]) for a in ("dz", "dy", "dx")] for r in rows])
    return (w[:, None] * v).sum(0) / w.sum()


def plot_displacements(res, out, offsets):
    rows = read_csv(os.path.join(res, "displacements.csv"))
    if not rows:
        return
    models = source_order(rows, "model")
    d = {m: np.array([[float(r[a]) for a in ("dz", "dy", "dx")] for r in rows if r["model"] == m]) for m in models}
    pos = {m: np.array([[float(r[a]) for a in ("gt_z", "gt_y", "gt_x")] for r in rows if r["model"] == m])
           for m in models}
    ideal = gt_offset(offsets)

    fig, axs = plt.subplots(len(models), 3, figsize=(10, 2.3 * len(models)), squeeze=False, sharex=True)
    bins = np.linspace(-6, 6, 49)
    for row, m in zip(axs, models):
        for k, ax in enumerate(row):
            v = d[m][:, k]
            ax.hist(v, bins=bins, color=AXIS_COLORS[k], edgecolor="white", linewidth=0.5)
            ax.axvline(0, color=MUTED, lw=0.8)
            ax.axvline(v.mean(), color=INK, lw=1.5)
            if ideal is not None:
                ax.axvline(ideal[k], color=INK, lw=1.2, ls="--")
            ax.set_title(f"{m}  d{AXES[k]}: mean {v.mean():+.2f}, n={len(v)}", loc="left")
    for ax, a in zip(axs[-1], AXES):
        ax.set_xlabel(f"pred - GT, {a} (voxels)")
    note = "solid = mean; dashed = GT's own offset (a perfect model lands here)" if ideal is not None else "solid = mean"
    fig.suptitle(note, x=0.01, ha="left", color=MUTED)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "displacement_hist.png"), dpi=130)
    plt.close(fig)

    fig, axs = plt.subplots(len(models), 3, figsize=(10, 2.3 * len(models)), squeeze=False)
    for row, m in zip(axs, models):
        for k, ax in enumerate(row):
            p, v = pos[m][:, k], d[m][:, k]
            ax.scatter(p, v, s=3, color=AXIS_COLORS[k], alpha=0.25, linewidths=0)
            edges = np.linspace(p.min(), p.max(), 9)
            idx = np.digitize(p, edges[1:-1])
            mids = [(edges[i] + edges[i + 1]) / 2 for i in range(8) if (idx == i).sum() > 10]
            means = [v[idx == i].mean() for i in range(8) if (idx == i).sum() > 10]
            ax.plot(mids, means, "o-", color=INK, lw=2, ms=4)
            ax.axhline(0, color=MUTED, lw=0.8)
            ax.set_ylim(-5, 5)
            ax.set_title(f"{m}: d{AXES[k]} vs GT {AXES[k]}", loc="left")
            ax.set_xlabel(f"GT {AXES[k]} (voxels)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "displacement_vs_position.png"), dpi=130)
    plt.close(fig)


def plot_per_tomogram(res, out):
    rows = read_csv(os.path.join(res, "per_tomogram.csv"))
    if not rows:
        return
    sources = source_order(rows, "source")
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    rng = np.random.default_rng(0)
    for k, ax in enumerate(axs):
        for i, s in enumerate(sources):
            v = np.array([float(r[("dz", "dy", "dx")[k]]) for r in rows if r["source"] == s])
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=12, color=SOURCE_COLORS[i % len(SOURCE_COLORS)],
                       alpha=0.7, linewidths=0)
            ax.plot([i - 0.3, i + 0.3], [np.median(v)] * 2, color=INK, lw=2)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks(range(len(sources)), sources)
        ax.set_title(f"density offset {AXES[k]} (voxels)", loc="left")
    axs[0].set_ylabel("density centre - coordinate")
    fig.suptitle("one dot per tomogram; bar = median", x=0.01, ha="left", color=MUTED)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "per_tomogram.png"), dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", "-r", required=True, help="Output folder of measure_pick_offset.py.")
    ap.add_argument("--output_dir", "-o", default=None, help="Defaults to <results_dir>/figures.")
    args = ap.parse_args()

    out = args.output_dir or os.path.join(args.results_dir, "figures")
    os.makedirs(out, exist_ok=True)
    offsets = read_csv(os.path.join(args.results_dir, "offsets.csv"))
    plot_averages(args.results_dir, out, offsets)
    plot_displacements(args.results_dir, out, offsets)
    plot_per_tomogram(args.results_dir, out)
    print(f"figures written to {out}")


if __name__ == "__main__":
    main()
