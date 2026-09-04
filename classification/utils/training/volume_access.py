"""Lazy subtomogram access.

``get_single_subtomogram`` loads the whole ~300 MB tomogram per crop, which is too slow for
contrastive pretraining. Zarr sources are sliced to the bounding box instead; other formats
fall back to a small per-process volume cache.

Handles live in a module-level cache so the datasets stay picklable -- torch_em serializes
``loader.dataset`` into every checkpoint.
"""

import os
from collections import OrderedDict
from typing import Optional, Tuple

import numpy as np
import zarr

from classification.config import MAX_EXTENT_HALO
from classification.training import get_volume

_ZARR_CACHE = {}
_ARRAY_CACHE = OrderedDict()
_ARRAY_CACHE_SIZE = 2


def bbox_size_for(max_extent: int, halo: int = MAX_EXTENT_HALO) -> int:
    """Crop size used by ``extract_subtomograms``; odd, so the particle stays centred."""
    size = max_extent + halo
    return size + 1 if size % 2 == 0 else size


def _find_zarr(path: str) -> Optional[str]:
    zarr_folders = [
        os.path.join(root, d)
        for root, dirs, _ in os.walk(path)
        for d in dirs
        if d.endswith(".zarr")
    ]
    if not zarr_folders:
        return None
    zarr_dir = next((f for f in zarr_folders if os.path.basename(f) == "denoised.zarr"), zarr_folders[0])
    level = os.path.join(zarr_dir, "0")
    return level if os.path.exists(level) else None


def _get_handle(path: str):
    """A lazily-sliceable array for `path`: a zarr array, or a cached numpy volume."""
    key = (os.getpid(), path)
    handle = _ZARR_CACHE.get(key)
    if handle is not None:
        return handle

    zarr_path = _find_zarr(path)
    if zarr_path is not None:
        handle = zarr.open(zarr_path, mode="r")
        _ZARR_CACHE[key] = handle
        return handle

    # Non-zarr formats have to be read in full; keep the last few around.
    if key in _ARRAY_CACHE:
        _ARRAY_CACHE.move_to_end(key)
        return _ARRAY_CACHE[key]
    volume = get_volume(path)
    _ARRAY_CACHE[key] = volume
    if len(_ARRAY_CACHE) > _ARRAY_CACHE_SIZE:
        _ARRAY_CACHE.popitem(last=False)
    return volume


def volume_shape(path: str) -> Tuple[int, int, int]:
    return tuple(_get_handle(path).shape)


def load_crop(
    path: str,
    coord: Tuple[int, int, int],
    max_extent: int,
    in_channels: int = 1,
    halo: int = MAX_EXTENT_HALO,
) -> np.ndarray:
    """One (C, D, H, W) crop, indexed exactly like ``extract_subtomograms``."""
    handle = _get_handle(path)
    half = bbox_size_for(max_extent, halo) // 2
    z, y, x = (int(round(v)) for v in coord)

    zmin, zmax = z - half, z + half + 1
    ymin, ymax = y - half, y + half + 1
    xmin, xmax = x - half, x + half + 1
    shape = handle.shape
    if not (zmin >= 0 and zmax <= shape[0] and ymin >= 0 and ymax <= shape[1] and xmin >= 0 and xmax <= shape[2]):
        raise IndexError(f"Crop at {coord} is out of bounds for volume of shape {shape}.")

    cube = np.asarray(handle[zmin:zmax, ymin:ymax, xmin:xmax], dtype=np.float32)
    cube = cube[None]
    if in_channels > 1:
        cube = np.repeat(cube, in_channels, axis=0)
    return cube
