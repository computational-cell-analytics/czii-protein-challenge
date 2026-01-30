import torch


class CryoETNormalize:
    def __init__(self, eps=1e-6, clip_percentile=0.01):
        self.eps = eps
        self.clip_percentile = clip_percentile

    def __call__(self, volume):
        #Ensure tensor
        if not isinstance(volume, torch.Tensor):
            volume = torch.tensor(volume, dtype=torch.float32)

        v = volume

        #clipping of outliers
        if self.clip_percentile is not None:
            flat = v.flatten()
            lo = torch.quantile(flat, self.clip_percentile)
            hi = torch.quantile(flat, 1.0 - self.clip_percentile)
            v = torch.clamp(v, lo, hi)

        mean = v.mean()
        std = v.std()

        v = (v - mean) / (std + self.eps)
        return v