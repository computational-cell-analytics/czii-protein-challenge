import os
from typing import Optional, Sequence, Union

import imageio.v3 as imageio
import numpy as np
from elf.io import open_file
from numpy.typing import ArrayLike

try:
    import tifffile
except ImportError:
    tifffile = None

TIF_EXTS = (".tif", ".tiff")
NON_CONTAINER_EXTS = (".mrc", ".mrcs", ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".nrrd", ".mha")


def supports_memmap(image_path):

    if tifffile is None:
        return False
    ext = os.path.splitext(image_path)[1]
    if ext.lower() not in TIF_EXTS:
        return False
    try:
        tifffile.memmap(image_path, mode="r")
    except ValueError:
        return False
    return True


def load_image(image_path, memmap=True):

    if supports_memmap(image_path) and memmap:
        return tifffile.memmap(image_path, mode="r")
    elif tifffile is not None and os.path.splitext(image_path)[1].lower() in TIF_EXTS:
        return tifffile.imread(image_path)
    elif os.path.splitext(image_path)[1].lower() == ".nrrd":
        import nrrd
        return nrrd.read(image_path)[0]
    elif os.path.splitext(image_path)[1].lower() == ".mha":
        import SimpleITK as sitk
        image = sitk.ReadImage(image_path)
        return sitk.GetArrayFromImage(image)
    else:
        return imageio.imread(image_path)


class MultiDatasetWrapper:

    def __init__(self, *file_datasets):
        # Make sure we have the same shapes.
        reference_shape = file_datasets[0].shape
        assert all(reference_shape == ds.shape for ds in file_datasets), "All datasets must have the same shape."
        self.file_datasets = file_datasets
        self.shape = (len(self.file_datasets),) + reference_shape

    def __getitem__(self, index):
        channel_index, spatial_index = index[:1], index[1:]
        data = [ds[spatial_index] for ds in self.file_datasets]
        data = np.stack(data)
        data = data[channel_index]
        return data


def _load_single_entry(path: str, key: Optional[str], mode: str):
    """Helper for loading one path that might be container or non-container."""
    ext = os.path.splitext(path)[1].lower()
    is_zarr = ext == ".zarr" or os.path.isdir(path) and any(f.endswith(".zarray") for f in os.listdir(path))

    if is_zarr or ext in (".n5", ".h5", ".hdf5"):
        # container-like
        with open_file(path, mode=mode) as f:
            if key is None:
                # Try to guess the first dataset
                datasets = list(f.keys())
                if not datasets:
                    raise ValueError(f"No datasets found in container {path}")
                key_to_use = datasets[0]
            else:
                key_to_use = key
            return f[key_to_use][:]
    else:
        # non-container (like .mrc, .tif, etc.)
        return load_image(path)


def load_data(
    path: Union[str, Sequence[str]],
    key: Optional[Union[str, Sequence[str]]] = None,
    mode: str = "r",
) -> ArrayLike:
    """Load data from a file or multiple files (.zarr, .mrc, etc.), with or without keys."""
    if isinstance(path, str):
        path = [path]
    if key is not None and isinstance(key, str):
        key = [key]
    
    # Load all entries (mix of zarr and mrc allowed)
    datasets = []
    for i, p in enumerate(path):
        current_key = None
        if key is not None:
            if len(key) == 1:
                current_key = key[0]
            elif len(key) == len(path):
                current_key = key[i]
            else:
                raise ValueError("If multiple keys are provided, their number must match the number of paths.")
        datasets.append(_load_single_entry(p, current_key, mode))

    # Stack if possible, or wrap
    shapes = [d.shape for d in datasets]
    if all(s == shapes[0] for s in shapes):
        return np.stack(datasets)
    else:
        return MultiDatasetWrapper(*datasets)
