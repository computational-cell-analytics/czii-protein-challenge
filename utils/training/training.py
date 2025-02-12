from typing import Optional, Tuple

import torch
import torch_em
from torch_em.model import AnisotropicUNet
import torch.nn as nn
# from torch_em.transform.augmentation import get_augmentations

from .data_loader import create_data_loader
from .heatmap_dataset import HeatmapDataset


'''#Julias code ... don't know yet if I need to chage it ...
def get_in_channels(image_path):
    # Load the first image to determine the number of channels
    image = np.asarray(imread(image_path))

    # Check if the first image is grayscale or RGB
    if len(image.shape) == 2:
        in_channels = 1
        # print(f"About to process grayscale images")
    elif image.shape[-1] == 4:
        in_channels = 3
        # print(f"About to process RGB images")
    else:
        in_channels = image.shape[-1]
        # print(f"About to process images of dimensions = {image.shape}")

    return in_channels'''


def get_3d_model(
    in_channels: int,
    out_channels: int,
    scale_factors: Tuple[Tuple[int, int, int]] = [[1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
    initial_features: int = 32,
    final_activation: Optional[str] = None,
) -> torch.nn.Module:
    """Get the 3D U-Net model.

    Args:
        in_channels: The number of input channels of the network.
        out_channels: The number of output channels of the network.
        scale_factors: The downscaling factors for each level of the U-Net encoder.
        initial_features: The number of features in the first level of the U-Net.
            The number of features increases by a factor of two in each level.
        final_activation: The activation applied to the last output layer.
    Returns:
        The U-Net.
    """
    model = AnisotropicUNet(
        scale_factors=scale_factors,
        in_channels=in_channels,
        out_channels=out_channels,
        initial_features=initial_features,
        gain=2,
        final_activation=final_activation,
    )
    return model


def supervised_training(
    name: str,
    train_paths: Tuple[str],
    train_label_paths: Tuple[str],
    val_paths: Tuple[str],
    val_label_paths: Tuple[str],
    raw_key: Optional[str],
    patch_shape: Tuple[int, int, int],
    batch_size: int = 1,
    lr: float = 1e-4,
    n_iterations: int = int(1e5),
    check: bool = False,
    out_channels: int = 2,
    augmentations: Optional[bool] = False,
    eps: float = 1e-5,
    sigma: int = None,
    lower_bound: float = None,
    upper_bound: float = None,
    test_paths: Optional[Tuple[str]] = None,
    test_label_paths: Optional[Tuple[str]] = None,
    save_root: Optional[str] = None,
    n_samples_train: Optional[int] = None,
    n_samples_val: Optional[int] = None,
    dataset_class=HeatmapDataset,
    **loader_kwargs,
):
    """
    Run supervised training

    Args:
        name: The name for the checkpoint to be trained.
        train_paths: Filepaths to the files for the training data. The files just contain the raw data.
        train_label_paths: Filepaths to the labels for the training data.
        val_paths: Filepaths to the files for the validation data.The files just contain the raw data.
        val_label_paths: Filepaths to the labels for the validation data.
        patch_shape: The patch shape used for a training example.
        batch_size: The batch size for training.
        lr: The initial learning rate.
        n_iterations: The number of iterations to train for.
        check: Whether to check the training and validation loaders instead of running training.
        out_channels: The number of output channels of the UNet.
        augmentations: Set to true if autmentations are needed.
        test_paths: Filepaths to the files for the test data.The files just contain the raw data.
        test_label_paths: Filepaths to the labels for the test data.
        save_root: Folder where the checkpoint will be saved.
        n_samples_train: The number of samples for the training dataset.
        n_samples_val: The number of samples for the validation dataset.
        dataset_class: The dataset class to use. By default `HeatmapDataset`, which creates a detection
            heatmap for the CZII Cryo Challenge data, is used.
        loader_kwargs: Additional keyword arguments for the dataloader.
    """
    if augmentations:
        # This is not implemented!
        raise NotImplementedError
        # raw_transform = DataAugmentations(p=0.25)
        # transform = get_augmentations(ndim=2)
    else:
        raw_transform = None
        transform = None

    num_workers = 6  # Julias example
    # num_workers = 0  # Julias example

    train_loader, val_loader, _ = create_data_loader(train_paths, train_label_paths,
                                                     val_paths, val_label_paths,
                                                     test_paths, test_label_paths,
                                                     raw_transform=raw_transform, transform=transform,
                                                     patch_shape=patch_shape, num_workers=num_workers,
                                                     batch_size=batch_size, raw_key=raw_key,
                                                     eps=eps, sigma=sigma,
                                                     lower_bound=lower_bound, upper_bound=upper_bound,
                                                     dataset_class=dataset_class,
                                                     n_samples_train=n_samples_train,
                                                     n_samples_val=n_samples_val)
    if check:
        from torch_em.util.debug import check_loader
        check_loader(train_loader, n_samples=4)
        check_loader(val_loader, n_samples=4)
        return

    in_channels = 1  # get_in_channels(train_images[0])
    model = get_3d_model(in_channels=in_channels, out_channels=out_channels)

    loss = nn.MSELoss(reduction="mean")
    metric = loss

    trainer = torch_em.default_segmentation_trainer(
        name=name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=lr,
        mixed_precision=True,
        log_image_interval=100,
        compile_model=False,
        save_root=save_root,
        loss=loss,
        metric=metric,
    )
    trainer.fit(n_iterations)
