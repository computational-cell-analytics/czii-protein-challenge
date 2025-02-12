from torch.utils.data import DataLoader
from .heatmap_dataset import HeatmapDataset
from torch_em.data.concat_dataset import ConcatDataset


def samples_to_datasets(n_samples, raw_paths, raw_key, split="uniform"):
    """@private
    """
    assert split in ("balanced", "uniform")
    n_datasets = len(raw_paths)
    if split == "uniform":
        # even distribution of samples to datasets
        samples_per_ds = n_samples // n_datasets
        divider = n_samples % n_datasets
        return [samples_per_ds + 1 if ii < divider else samples_per_ds for ii in range(n_datasets)]
    else:
        # distribution of samples to dataset based on the dataset lens
        raise NotImplementedError


def _load_dataset(
    raw_paths,
    label_paths,
    raw_transform, transform,
    patch_shape,
    raw_key=None,
    eps=0.00001, sigma=None,
    lower_bound=None, upper_bound=None,
    dataset_class=HeatmapDataset,
    n_samples=None,
):

    if isinstance(raw_paths, str):
        ds = dataset_class(
            raw_path=raw_paths, raw_key=raw_key, label_path=label_paths, patch_shape=patch_shape,
            raw_transform=raw_transform, transform=transform, eps=eps, sigma=sigma,
            lower_bound=lower_bound, upper_bound=upper_bound, n_samples=n_samples,
        )
    else:
        assert len(raw_paths) > 0

        samples_per_ds = (
            [None] * len(raw_paths) if n_samples is None else samples_to_datasets(n_samples, raw_paths, raw_key)
        )
        ds = []
        for i, (raw_path, label_path) in enumerate(zip(raw_paths, label_paths)):

            dset = dataset_class(
                raw_path=raw_path, raw_key=raw_key, label_path=label_path, patch_shape=patch_shape,
                raw_transform=raw_transform, transform=transform, eps=eps, sigma=sigma,
                lower_bound=lower_bound, upper_bound=upper_bound, n_samples=samples_per_ds[i],
                )

            ds.append(dset)
        ds = ConcatDataset(*ds)
    return ds


def create_data_loader(
    train_images, train_labels,
    val_images, val_labels,
    test_images, test_labels,
    raw_transform, transform,
    patch_shape, num_workers, batch_size,
    raw_key=None,
    eps=0.00001, sigma=None,
    lower_bound=None, upper_bound=None,
    dataset_class=HeatmapDataset,
    n_samples_train=None,
    n_samples_val=None,
):
    train_set = _load_dataset(
        raw_paths=train_images, raw_key=raw_key, label_paths=train_labels, patch_shape=patch_shape,
        raw_transform=raw_transform, transform=transform, eps=eps, sigma=sigma,
        lower_bound=lower_bound, upper_bound=upper_bound, n_samples=n_samples_train,
        dataset_class=dataset_class,
    )
    val_set = _load_dataset(
        raw_paths=val_images, raw_key=raw_key, label_paths=val_labels, patch_shape=patch_shape,
        raw_transform=raw_transform, transform=transform, eps=eps, sigma=sigma,
        lower_bound=lower_bound, upper_bound=upper_bound, n_samples=n_samples_val,
        dataset_class=dataset_class,
    )
    test_set = _load_dataset(
        raw_paths=test_images, raw_key=raw_key, label_paths=test_labels, patch_shape=patch_shape,
        raw_transform=raw_transform, transform=transform, eps=eps, sigma=sigma,
        lower_bound=lower_bound, upper_bound=upper_bound,
        dataset_class=dataset_class,
    )

    # put into DataLoader
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers)
    val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=True,
                                 num_workers=num_workers)

    train_dataloader.shuffle = True
    val_dataloader.shuffle = True
    test_dataloader.shuffle = True

    return train_dataloader, val_dataloader, test_dataloader
