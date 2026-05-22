import numpy as np

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
