from typing import Sequence, Tuple, Callable
import numpy as np
import torch
from numpy.typing import ArrayLike
from skimage.transform import resize

from classification.data_processing import extract_subtomograms


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
        subtomogram: Sequence[ArrayLike],
        target: Sequence,
        normalization: Callable = None,
        augmentation: Callable = None,
        image_shape: Tuple[int, int, int] = None,
        n_classes: int = 2,
        n_samples: int = None,
    ):
        self.normalization = normalization
        self.augmentation = augmentation
        self.image_shape = image_shape
        self.n_classes = n_classes

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

