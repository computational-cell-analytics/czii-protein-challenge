import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import Tuple, Dict

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

    # text box with labels and probs
    prob_text = f"Prob: {pred_prob:.3f}" if pred_prob is not None else ""
    info_lines = [f"ID: {sample_id}", f"Pred: {pred_label}", f"GT: {true_label}", prob_text]
    if other_info:
        for k, v in other_info.items():
            info_lines.append(f"{k}: {v}")

    fig.suptitle("\n".join(info_lines), fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(outpath, dpi=150)
    plt.close(fig)
    return outpath