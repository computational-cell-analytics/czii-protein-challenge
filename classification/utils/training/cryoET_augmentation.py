import torch
import torch.nn.functional as F
import random
import math

#proteins have no orientation, so should use random rotation
def random_rot90(volume):
    #volume: (1, D, H, W)
    k = random.randint(0, 3)
    axis = random.choice([(2,3), (1,3), (1,2)])
    return torch.rot90(volume, k, axis)

#random flips only along x any y axis because the missing wedge could be asymmetric (#TODO could check how it is in our case)
def random_flip(volume):
    if random.random() < 0.5:
        volume = torch.flip(volume, dims=[2])  # Y right?
    if random.random() < 0.5:
        volume = torch.flip(volume, dims=[3])  # X right?
    return volume

#TODO maybe unnecessary?
def add_gaussian_noise(volume, sigma_range=(0.0, 0.1)):
    sigma = random.uniform(*sigma_range)
    noise = torch.randn_like(volume) * sigma
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


def random_intensity(volume, scale_range=(0.9, 1.1), shift_range=(-0.1, 0.1)):
    #scale for different intensities; >1 for intensity amplification and <1 for compression; can simuplate different exposures in the imaging process
    scale = random.uniform(*scale_range)
    #shift to add offset; >0 to get brighter background, <0 to get darker; can simulate ice thickness variation
    shift = random.uniform(*shift_range)
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

        if random.random() < 0.7:
            volume = add_gaussian_noise(volume)
            info.append("noise")

        if random.random() < 0.3:
            volume = gaussian_blur_3d(volume)
            info.append("blur")

        if random.random() < 0.5:
            volume = random_intensity(volume)
            info.append("intensity")

        if return_info:
            return volume, info

        return volume
