

######TODO####################################
#implement augmentations that also move the 'peak'/coordinate of the protein a little. I think I want it here

from typing import Callable, List, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_em.data.concat_dataset import ConcatDataset
from numpy.typing import ArrayLike

from .classification_dataset import ClassificationDataset
from sklearn.preprocessing import LabelEncoder

def samples_to_datasets(n_samples: int, n_datasets: int, split: str = "uniform") -> List[int]:
    assert split in ("balanced", "uniform")
    if split == "uniform":
        samples_per_ds = n_samples // n_datasets
        remainder = n_samples % n_datasets
        return [samples_per_ds + 1 if i < remainder else samples_per_ds for i in range(n_datasets)]
    raise NotImplementedError("Balanced splitting is not implemented.")

def _load_dataset(
    paths: List[str],
    zarr_: bool = True,
    in_channels: int = 1,
    max_extent:int = 39,
    target_root: str = None,
    image_shape: Tuple[int, int, int] = None,
    normalization: Callable = None,
    augmentation: Callable = None,
    dataset_class=ClassificationDataset,
    n_samples: Union[int, None] = None,
    n_classes: int = 2,
) -> torch.utils.data.Dataset:
    """
    Load subtomograms and their corresponding labels into a single Dataset.

    Args:
        paths: A List of paths to the full tomograms
        zarr_: says if the full tomogram is a omezarr file or not
        max_extent: size of bbox for the subtomograms of the proteins
        image_shape: Target shape to resize subtomograms (D, H, W).
        normalization: Function to normalize each subtomogram.
        augmentation: Function to augment each subtomogram.
        dataset_class: Dataset class to use (default: ClassificationDataset).
        n_classes: Number of classes for classification.

    Returns:
        A single Dataset containing all subtomograms and labels.
    """

    n_samples = len(paths)

    print(f"n_samples {n_samples}")

    ds = dataset_class(
        paths=paths,
        zarr_=zarr_,
        in_channels=in_channels,
        max_extent=max_extent,
        target_root=target_root,
        normalization=normalization,
        augmentation=augmentation,
        image_shape=image_shape,
        n_classes=n_classes,
        n_samples=n_samples,
    )


    #TODO more flexible along different datasets?
    '''
    if isinstance(subtomogram_list, np.ndarray):
        ds = dataset_class(
            subtomogram=subtomogram_list[None, ...],  # wrap in a list to keep consistent shape
            target=targets,
            normalization=normalization,
            augmentation=augmentation,
            image_shape=image_shape,
            n_classes=n_classes,
            n_samples=n_samples,
        )
    else:
        samples_per_ds = (
            [None] * len(subtomogram_list) if n_samples is None else samples_to_datasets(n_samples, len(subtomogram_list))
        )
        ds = []
        for i, (subtomogram_list, targets) in enumerate(zip(subtomogram_list, targets)):

            dset = dataset_class(
                subtomogram=subtomogram_list,
                target=targets,
                normalization=normalization,
                augmentation=augmentation,
                image_shape=image_shape,
                n_classes=n_classes,
                n_samples=samples_per_ds[i],
            )

            ds.append(dset)
        ds = ConcatDataset(*ds)
    '''

    return ds


def create_data_loader(
    train_data: List[str],
    val_data: List[str],
    test_data: List[str],
    zarr_: bool = True,
    in_channels: int = 1,
    max_extent:int = 39,
    target_root: str = None,
    normalization: Callable = None,
    augmentation: Callable = None,
    patch_shape: Tuple[int, int, int] = (32, 32, 32),
    num_workers: int = 4,
    batch_size: int = 8,
    dataset_class=ClassificationDataset,
    n_samples_train: Union[int, None] = None,
    n_samples_val: Union[int, None] = None,
    n_classes: int = 2,
):

    train_set = _load_dataset(
        train_data, 
        zarr_, in_channels, max_extent, target_root,
        patch_shape,
        normalization, augmentation,
        dataset_class, n_samples_train, n_classes
    )
    
    val_set = _load_dataset(
        val_data, 
        zarr_, in_channels, max_extent, target_root,
        patch_shape,
        normalization, augmentation,
        dataset_class, n_samples_val, n_classes
    )

    test_set = _load_dataset(
        test_data,
        zarr_, in_channels, max_extent, target_root,
        patch_shape,
        normalization, augmentation,
        dataset_class, n_classes=n_classes
    )

    # --- Encode string labels to integers ---
    all_labels = np.concatenate([train_set.targets, val_set.targets])  # <--- FIXED
    encoder = LabelEncoder()
    encoder.fit(all_labels)

    train_target_int = encoder.transform(train_set.targets)
    val_target_int = encoder.transform(val_set.targets)
    test_target_int = encoder.transform(test_set.targets) if test_set is not None else None  # <--- FIXED

    idx_to_label = {i: label for i, label in enumerate(encoder.classes_)}

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)


    train_loader.shuffle = True
    val_loader.shuffle = True
    test_loader.shuffle=True

    return train_loader, val_loader, test_loader, idx_to_label


