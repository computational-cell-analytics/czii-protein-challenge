from typing import List, Tuple, Callable
import numpy as np
import torch
from skimage.transform import resize

from itertools import chain
from classification.training import get_coords_and_targets, get_single_subtomogram, get_volume

import os
import h5py
from torch.utils.data import get_worker_info

class ClassificationDataset(torch.utils.data.Dataset):
    """
    Dataset for classification training using lazy subtomogram extraction.
    Automatically skips out-of-bounds subtomograms.
    """

    def __init__(
        self,
        paths: List[str],
        in_channels: int = 1,
        max_extent: int = 39,
        target_root: str = None,
        normalization: Callable = None,
        augmentation: Callable = None,
        image_shape: Tuple[int, int, int] = None,
        n_classes: int = 2,
        n_samples: int = None,
    ):
        self.in_channels = in_channels
        self.max_extent = max_extent
        self.target_root = target_root
        self.normalization = normalization
        self.augmentation = augmentation
        self.image_shape = image_shape
        self.n_classes = n_classes

        # Load metadata per tomogram
        coords_list, targets_list = get_coords_and_targets(paths, target_root=self.target_root)

        # Filter out-of-bounds coordinates
        valid_coords = []
        valid_targets = []
        valid_paths = []

        for path, coords, targets in zip(paths, coords_list, targets_list):
            volume = get_volume(path)
            D, H, W = volume.shape
            half = self.max_extent // 2
            for c, t in zip(coords, targets):
                z, y, x = c
                if (z - half >= 0 and z + half < D and
                    y - half >= 0 and y + half < H and
                    x - half >= 0 and x + half < W):
                    valid_coords.append(c)
                    valid_targets.append(t)
                    valid_paths.append(path)

        self.coords = valid_coords
        self.targets = valid_targets
        self.paths = valid_paths

        # Label mapping
        self.classes = sorted(set(self.targets))
        self.label_to_index = {label: idx for idx, label in enumerate(self.classes)}

        # Check n_samples if specified
        if n_samples is not None and len(self.targets) != n_samples:
            raise ValueError(f"Expected {n_samples} samples, got {len(self.targets)}")

        print(f"Initialized dataset with {len(self.targets)} subtomograms from {len(paths)} tomograms")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        path = self.paths[index]
        coord = self.coords[index]
        label = self.targets[index]

        try:
            x, y = get_single_subtomogram(
                path,
                coord,
                self.max_extent,
                self.in_channels,
                label,
            )
        except (IndexError, ValueError):
            import random
            new_index = random.randint(0, len(self) - 1)
            return self[new_index]

        #Normalization
        if self.normalization is not None:
            x = self.normalization(x)

        #Resize
        if self.image_shape is not None:
            x = self._resize(x)
        
        #Convert to tensors
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        #Augmentation
        if self.augmentation is not None:
            _shape = x.shape
            x, aug_info = self.augmentation(x, return_info=True)
            assert x.shape == _shape

            #Just for checking the augmentations, will delete later:
            worker_info = get_worker_info()
            is_main_worker = (worker_info is None) or (worker_info.id == 0)

            if is_main_worker:
                save_dir = "./_debug_augmented_examples"
                os.makedirs(save_dir, exist_ok=True)

                if not hasattr(self, "_debug_save_count"):
                    self._debug_save_count = 0

                if self._debug_save_count < 25:
                    aug_tag = "none" if len(aug_info) == 0 else "+".join(aug_info)
                    fname = f"sample_{self._debug_save_count:03d}_aug_{aug_tag}.h5"
                    h5_path = os.path.join(save_dir, fname)

                    with h5py.File(h5_path, "w") as f:
                        f.create_dataset(
                            "x",
                            data=x.detach().cpu().numpy(),
                            compression="gzip",
                        )

                    self._debug_save_count += 1


        

        if isinstance(y, str):
            y = self.label_to_index[y]
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.long)

        return x, y



    def _resize(self, x):
        out = [resize(ch, self.image_shape, preserve_range=True)[None] for ch in x]
        return np.concatenate(out, axis=0)

    @property
    def ndim(self):
        # Lazily compute ndim from first sample
        sample, _ = self[0]
        return sample.ndim

    @property
    def targets_array(self):
        return np.array(self.targets)
