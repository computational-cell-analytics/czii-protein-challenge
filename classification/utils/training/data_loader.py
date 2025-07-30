

######TODO####################################
#implement augmentations that also move the 'peak'/coordinate of the protein a little. I think I want it here

from typing import Callable, List, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_em.data.concat_dataset import ConcatDataset

from .classification_dataset import ClassificationDataset
from ...data_processing.create_subtomograms import get_max_extent


def samples_to_datasets(n_samples: int, n_datasets: int, split: str = "uniform") -> List[int]:
    assert split in ("balanced", "uniform")
    if split == "uniform":
        samples_per_ds = n_samples // n_datasets
        remainder = n_samples % n_datasets
        return [samples_per_ds + 1 if i < remainder else samples_per_ds for i in range(n_datasets)]
    raise NotImplementedError("Balanced splitting is not implemented.")


def compute_max_extent_from_all(
    all_peaks: List[Sequence[Tuple[int, int, int]]],
    all_heatmaps: List[np.ndarray],
) -> Tuple[int, int, int]:
    max_extents = [
        get_max_extent(peaks, heatmap)
        for peaks, heatmap in zip(all_peaks, all_heatmaps)
    ]
    max_extent = max(max_extents)
    return (max_extent, max_extent, max_extent)


def _load_dataset(
    raw_data_list: List[np.ndarray],
    peaks_list: List[Sequence[Tuple[int, int, int]]],
    targets_list: List[Sequence],
    image_shape: Tuple[int, int, int] = None,
    normalization: Callable = None,
    augmentation: Callable = None,
    dataset_class=ClassificationDataset,
    n_samples: Union[int, None] = None,
) -> torch.utils.data.Dataset:
    assert len(raw_data_list) == len(peaks_list) == len(targets_list), "Input lists must match in length"

    n_datasets = len(raw_data_list)
    samples_per_ds = [None] * n_datasets if n_samples is None else samples_to_datasets(n_samples, n_datasets)

    datasets = []
    for i in range(n_datasets):
        dataset = dataset_class(
            raw_data=raw_data_list[i],
            peaks=peaks_list[i],
            target=targets_list[i],
            max_extent=image_shape,
            normalization=normalization,
            augmentation=augmentation,
        )
        datasets.append(dataset)

    return datasets[0] if len(datasets) == 1 else ConcatDataset(*datasets)


def create_data_loader(
    train_data: Tuple[List[np.ndarray], List[Sequence[Tuple[int, int, int]]], List[np.ndarray], List[Sequence]],
    val_data: Tuple[List[np.ndarray], List[Sequence[Tuple[int, int, int]]], List[np.ndarray], List[Sequence]],
    test_data: Tuple[List[np.ndarray], List[Sequence[Tuple[int, int, int]]], List[np.ndarray], List[Sequence]],
    normalization: Callable = None,
    augmentation: Callable = None,
    patch_shape: Tuple[int, int, int] = None,
    num_workers: int = 4,
    batch_size: int = 8,
    dataset_class=ClassificationDataset,
    n_samples_train: Union[int, None] = None,
    n_samples_val: Union[int, None] = None,
):
    train_raws, train_peaks, train_heatmaps, train_targets = train_data
    val_raws, val_peaks, val_heatmaps, val_targets = val_data
    test_raws, test_peaks, test_heatmaps, test_targets = test_data

    # Compute max_extent over all datasets
    all_peaks = train_peaks + val_peaks + test_peaks
    all_heatmaps = train_heatmaps + val_heatmaps + test_heatmaps
    max_extent = compute_max_extent_from_all(all_peaks, all_heatmaps)

    # Use user-specified patch_shape if provided, else use computed max_extent
    final_patch_shape = patch_shape if patch_shape is not None else max_extent

    train_set = _load_dataset(
        train_raws, train_peaks, train_targets, final_patch_shape,
        normalization, augmentation,
        dataset_class, n_samples_train
    )

    val_set = _load_dataset(
        val_raws, val_peaks, val_targets, final_patch_shape,
        normalization, augmentation,
        dataset_class, n_samples_val
    )

    test_set = _load_dataset(
        test_raws, test_peaks, test_targets, final_patch_shape,
        normalization, augmentation,
        dataset_class
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

