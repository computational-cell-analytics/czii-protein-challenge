from typing import Sequence, Tuple, Callable, List
import numpy as np
import torch
from numpy.typing import ArrayLike
from skimage.transform import resize

from classification.data_processing import extract_subtomograms
from classification.training import get_coords_and_targets, get_data

class ClassificationDataset(torch.utils.data.Dataset):
    """
    Dataset for classification training using pre-extracted 3D subtomograms and their labels.

    Args:
        subtomogram (Sequence[ArrayLike]): 
            Sequence of 3D subtomogram volumes (e.g., numpy arrays with shape (C, D, H, W)).
        target (Sequence):
            Sequence of classification labels corresponding to each subtomogram.
        normalization (Callable, optional):
            Function to apply normalization to each subtomogram.
        augmentation (Callable, optional):
            Function to apply data augmentation to each subtomogram.
        image_shape (Tuple[int, int, int], optional):
            If given, each subtomogram will be resized to this shape (D, H, W).
        n_classes (int, optional):
            Number of output classes. Used for validation or consistency checks.
        n_samples (int, optional):
            Expected number of samples. If provided, will be checked against the length of the data.

    Returns:
        (ndarray, label): Tuple containing the processed subtomogram and its corresponding label.
    """
    def __init__(
        self,
        paths: List[str],
        zarr_: bool = True,
        in_channels: int = 1,
        max_extent: int = 39,
        target_root: str = None,
        normalization: Callable = None,
        augmentation: Callable = None,
        image_shape: Tuple[int, int, int] = None,
        n_classes: int = 2,
        n_samples: int = None,
    ):
        self.target_root = target_root
        self.normalization = normalization
        self.augmentation = augmentation
        self.image_shape = image_shape
        self.n_classes = n_classes

        print(f"max_extent {max_extent}")
        #TODO maybe also put this into dataset? maybe not needed cuz it doesnt take too much memory
        coords, target_ = get_coords_and_targets(paths, target_root=self.target_root)
        
        # Now extract subtomograms
        #TODO can I include the augmentation with the coordinate being slightly off in get_data???
        subtomogram, target = get_data(paths, coords, max_extent, in_channels=in_channels, targets=target_, zarr_=zarr_)

        self.data = subtomogram
        self.target = target

        if len(self.data) != len(self.target):
            raise ValueError(f"Length of data and target don't agree: {len(self.data)} != {len(self.target)}")

        # Create mapping from string labels to integer indices
        self.classes = sorted(list(set(self.target)))
        self.label_to_index = {label: idx for idx, label in enumerate(self.classes)}


    def __len__(self):
        return len(self.data)

    def resize(self, x):
        """@private
        """
        out = [resize(channel, self.image_shape, preserve_range=True)[None] for channel in x]
        return np.concatenate(out, axis=0)

    def __getitem__(self, index):
        x, y = self.data[index], self.target[index]

        # apply normalization
        if self.normalization is not None:
            x = self.normalization(x)

        # resize to sample shape if it was given
        if self.image_shape is not None:
            x = self.resize(x)

        # apply augmentations (if any)
        if self.augmentation is not None:
            _shape = x.shape
            # adds unwanted batch axis
            x = self.augmentation(x)[0][0]
            assert x.shape == _shape


        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        if isinstance(y, str):
            y = self.label_to_index[y]
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.long)

        return x, y


    @property
    def ndim(self):
        return self.data[0].ndim
    
    @property
    def targets(self):
        return self.target

