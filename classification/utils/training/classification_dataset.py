from typing import Sequence, Tuple, Callable
import numpy as np
import torch
from numpy.typing import ArrayLike
from skimage.transform import resize

from ...data_processing.create_subtomograms import extract_subtomograms


class ClassificationDataset(torch.utils.data.Dataset):
    """
    Dataset for classification training using subtomograms.

    Args:
        raw_data: 3D tomogram input (numpy array)
        peaks: list of coordinates for subtomogram extraction
        max_extent: size used to extract subtomograms
        normalization: function applied to each subtomogram (optional)
        augmentation: data augmentation function (optional)
        image_shape: tuple to resize each subtomogram (optional)
    """
    def __init__(
        self,
        raw_data: np.ndarray,
        peaks: Sequence[Tuple[int, int, int]],
        max_extent: int,
        normalization: Callable = None,
        augmentation: Callable = None,
        image_shape: Tuple[int, int, int] = None,
        n_classes: int = 2,
    ):

        self.normalization = normalization
        self.augmentation = augmentation
        self.image_shape = image_shape
        self.n_classes = n_classes

        self.data = extract_subtomograms(raw_data, peaks, max_extent)

        n_samples = len(peaks)
        self.target = list(np.random.randint(0, self.n_classes, size=n_samples))

        if len(peaks) != len(self.target):
            raise ValueError(f"Length of peaks and target don't agree: {len(peaks)} != {len(self.target)}")


    def __len__(self):
        return len(self.data)

    def resize(self, x: np.ndarray) -> np.ndarray:
        """Resize 3D volume (subtomogram) to target shape."""
        return resize(x, self.image_shape, preserve_range=True)

    def __getitem__(self, index: int):
        x, y = self.data[index], self.target[index]

        if self.normalization is not None:
            x = self.normalization(x)

        if self.image_shape is not None:
            x = self.resize(x)

        if self.augmentation is not None:
            original_shape = x.shape
            x = self.augmentation(x)[0][0]
            assert x.shape == original_shape

        return x, y

    @property
    def ndim(self):
        return self.data[0].ndim

