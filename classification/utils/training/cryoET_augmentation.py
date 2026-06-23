import torch
import torch.nn.functional as F
import random
import math


#proteins have no orientation, so should use random rotation
def random_rot90(volume):
    #volume: (1, D, H, W)
    k = random.randint(1, 3)  # exclude 0, always rotate along x-y axis
    return torch.rot90(volume, k, dims=(2, 3))


#random flips only along x any y axis because the missing wedge could be asymmetric 
def random_flip(volume):
    if random.random() < 0.5:
        volume = torch.flip(volume, dims=[2])  # Y 
    if random.random() < 0.5:
        volume = torch.flip(volume, dims=[3])  # X 
    return volume


#add noise relative to Mean squared voxel intensity with a range
def add_gaussian_noise(volume, snr_db_range=(20, 30)):
    signal_power = volume.std().pow(2)
    snr_db = random.uniform(*snr_db_range)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = torch.randn_like(volume) * torch.sqrt(noise_power)
    return volume + noise


#using small sigma to not change the resolution too much
def gaussian_blur_3d(volume, sigma_range=(0.3, 1.0)):
    sigma = random.uniform(*sigma_range)
    size = int(2 * math.ceil(2 * sigma) + 1)

    coords = torch.arange(size) - size // 2
    grid = torch.stack(torch.meshgrid(coords, coords, coords), dim=-1)
    kernel = torch.exp(-grid.pow(2).sum(-1) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()

    kernel = kernel.view(1, 1, size, size, size).to(volume.device)
    volume = F.conv3d(volume.unsqueeze(0), kernel, padding=size//2)
    return volume.squeeze(0)


def _center_crop_or_pad(volume, target_shape):
    #volume: (C, D, H, W); target_shape: (D, H, W)
    #center-crop the dimensions that are too large
    slices = [slice(None)]
    for cur, tgt in zip(volume.shape[1:], target_shape):
        if cur > tgt:
            start = (cur - tgt) // 2
            slices.append(slice(start, start + tgt))
        else:
            slices.append(slice(None))
    volume = volume[tuple(slices)]

    #center-pad the dimensions that are too small (replicate edges to avoid hard borders)
    pad = []
    for cur, tgt in zip(volume.shape[1:], target_shape):  # D, H, W order
        if cur < tgt:
            total = tgt - cur
            before = total // 2
            pad.append((before, total - before))
        else:
            pad.append((0, 0))
    #F.pad expects the last spatial dim first: (W_before, W_after, H_..., D_...)
    pad_arg = [v for bef_aft in reversed(pad) for v in bef_aft]
    if any(pad_arg):
        volume = F.pad(volume.unsqueeze(0), pad_arg, mode="replicate").squeeze(0)

    return volume


#random isotropic rescaling to simulate pixel-/particle-size variation (as in easymode);
#>1 zooms in (particles appear larger), <1 zooms out (smaller). The volume is rescaled and
#then cropped/padded back to its original shape so the particle stays centered.
def random_rescale(volume, scale_range=(0.9, 1.1)):
    #volume: (C, D, H, W)
    scale = random.uniform(*scale_range)
    c, d, h, w = volume.shape
    new_size = (
        max(1, int(round(d * scale))),
        max(1, int(round(h * scale))),
        max(1, int(round(w * scale))),
    )

    rescaled = F.interpolate(
        volume.unsqueeze(0),
        size=new_size,
        mode="trilinear",
        align_corners=False,
    ).squeeze(0)

    return _center_crop_or_pad(rescaled, (d, h, w))


def random_intensity(
    volume,
    scale_range=(0.95, 1.05),
    shift_std_fraction=(0.0, 0.1),
):
    #scale for different intensities; >1 for intensity amplification and <1 for compression; can simuplate different exposures in the imaging process
    scale = random.uniform(*scale_range)
    #shift to add offset; >0 to get brighter background, <0 to get darker; can simulate ice thickness variation
    shift_sigma = volume.std()
    shift = torch.randn(1, device=volume.device) * shift_sigma \
            * random.uniform(*shift_std_fraction)

    return volume * scale + shift


class CryoETAugment:
    def __init__(self):
        pass

    def __call__(self, volume, return_info=False):
        info = []

        if random.random() < 0.9:
            volume = random_rot90(volume)
            info.append("rot90")

        if random.random() < 0.5:
            volume = random_flip(volume)
            info.append("flip")

        if random.random() < 0.2:
            volume = add_gaussian_noise(volume)
            info.append("noise")

        if random.random() < 0.2:
            volume = gaussian_blur_3d(volume)
            info.append("blur")

        if random.random() < 0.4:
            volume = random_intensity(volume)
            info.append("intensity")

        if random.random() < 0.3:
            volume = random_rescale(volume)
            info.append("rescale")

        if return_info:
            return volume, info

        return volume
