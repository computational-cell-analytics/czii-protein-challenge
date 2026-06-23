import numpy as np
import torch
import torch.nn.functional as F

from .detection_dataset import robust_standardize


class ContrastAugmentation:
    """Randomly varies intensity/contrast before robust standardization.

    Applies two independent augmentations to make the network invariant to
    dark pigmented regions and varying contrast conditions:
    - Random gamma: non-linear remapping that boosts or suppresses dark voxels
    - Random percentile clipping: simulates different acquisition contrasts
    """

    def __init__(self, p=0.5, gamma_range=(0.5, 2.0), clip_percentile_range=(0.5, 5.0), eps=1e-8):
        self.p = p
        self.gamma_range = gamma_range
        self.clip_percentile_range = clip_percentile_range
        self.eps = eps

    def __call__(self, raw):
        raw = raw.astype(np.float32)

        # Random gamma augmentation
        if np.random.random() < self.p:
            gamma = np.random.uniform(*self.gamma_range)
            raw_min, raw_max = raw.min(), raw.max()
            if raw_max > raw_min:
                raw_norm = (raw - raw_min) / (raw_max - raw_min + self.eps)
                raw = raw_min + (raw_max - raw_min) * (raw_norm ** gamma)

        # Random percentile clipping (simulates varying acquisition contrast)
        if np.random.random() < self.p:
            low_pct = np.random.uniform(0.0, self.clip_percentile_range[1])
            high_pct = np.random.uniform(100.0 - self.clip_percentile_range[1], 100.0)
            low_val = np.percentile(raw, low_pct)
            high_val = np.percentile(raw, high_pct)
            raw = np.clip(raw, low_val, high_val)

        return robust_standardize(raw, eps=self.eps)


def _center_crop_or_pad_3d(tensor, target_shape):
    """Center-crop (oversized dims) or edge-pad (undersized dims) a (C, Z, Y, X)
    tensor back to a target spatial shape (Z, Y, X)."""
    # center-crop the dimensions that are too large
    slices = [slice(None)]
    for cur, tgt in zip(tensor.shape[1:], target_shape):
        if cur > tgt:
            start = (cur - tgt) // 2
            slices.append(slice(start, start + tgt))
        else:
            slices.append(slice(None))
    tensor = tensor[tuple(slices)]

    # center-pad the dimensions that are too small (replicate edges)
    pad = []
    for cur, tgt in zip(tensor.shape[1:], target_shape):  # Z, Y, X order
        if cur < tgt:
            total = tgt - cur
            before = total // 2
            pad.append((before, total - before))
        else:
            pad.append((0, 0))
    # F.pad expects the last spatial dim first: (X_before, X_after, Y_..., Z_...)
    pad_arg = [v for bef_aft in reversed(pad) for v in bef_aft]
    if any(pad_arg):
        tensor = F.pad(tensor.unsqueeze(0), pad_arg, mode="replicate").squeeze(0)

    return tensor


class RandomRescale:
    """Randomly rescales (zooms) a patch and its labels by a common factor.

    Simulates pixel-/particle-size variation (as in easymode). A factor >1 enlarges
    particles (zoom in), <1 shrinks them (zoom out). Raw and labels are rescaled
    jointly so the heatmap/flow stay aligned, then center-cropped or edge-padded back
    to the original patch shape so the geometry remains consistent.

    This is a joint (raw, labels) transform and must be used as the dataset `transform`
    (which receives both), not as a `raw_transform` (which only sees the raw).

    Note: the stereographic flow encodes directions, which are preserved under
    isotropic scaling. Its distance decay (set by `sigma`) is only approximately
    rescaled, which is acceptable for augmentation.
    """

    def __init__(self, p=0.5, scale_range=(0.9, 1.1)):
        self.p = p
        self.scale_range = scale_range

    def __call__(self, raw, labels):
        if np.random.random() >= self.p:
            return raw, labels

        scale = float(np.random.uniform(*self.scale_range))
        raw = self._rescale(raw, scale, has_channels=(raw.ndim == 4))
        labels = self._rescale(labels, scale, has_channels=True)
        return raw, labels

    @staticmethod
    def _rescale(arr, scale, has_channels):
        out_dtype = arr.dtype
        t = torch.as_tensor(np.ascontiguousarray(arr), dtype=torch.float32)
        if not has_channels:
            t = t.unsqueeze(0)  # add a channel dim -> (C, Z, Y, X)

        spatial = tuple(t.shape[1:])
        new_spatial = [max(1, int(round(s * scale))) for s in spatial]

        t = F.interpolate(
            t.unsqueeze(0), size=new_spatial, mode="trilinear", align_corners=False
        ).squeeze(0)
        t = _center_crop_or_pad_3d(t, spatial)

        out = t.numpy()
        if not has_channels:
            out = out[0]
        return np.ascontiguousarray(out, dtype=out_dtype)
