import warnings
from functools import partial
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
import sklearn.metrics as metrics
import torch
import torch_em

from torch_em.classification.classification_logger import ClassificationLogger
from torch_em.classification.classification_trainer import ClassificationTrainer

from .data_loader import create_data_loader
from .classification_dataset import ClassificationDataset

from torch_em.model.resnet3d import resnet3d_18

class ClassificationMetric:
    """Metric for classification training.

    Args:
        metric_name: The name of the metric. Must be a valid sklearn.metrics function.
        metric_kwargs: Keyword arguments for the metric.
    """

    def __init__(self, metric_name: str = "accuracy_score", **metric_kwargs):
        if not hasattr(metrics, metric_name):
            raise ValueError(f"Invalid metric_name '{metric_name}'. Must be from sklearn.metrics.")
        self.metric = getattr(metrics, metric_name)
        self.metric_kwargs = metric_kwargs

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Evaluate model prediction against classification labels."""
        return 1.0 - self.metric(y_true, y_pred, **self.metric_kwargs)


def get_3d_model(
    in_channels: int,
    out_channels: int,
) -> torch.nn.Module:
    """Get the 3D ResNet model.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels (i.e., number of classes).
    """
    # TODO: Implement EfficientNet 
    #TODO add more arguments?
    model = resnet3d_18(in_channels=in_channels, out_channels=out_channels)
    return model


def classification_training(
    name: str,
    train_data: Tuple[
        List[np.ndarray], List[Sequence[Tuple[int, int, int]]], List[np.ndarray]
    ],
    val_data: Tuple[
        List[np.ndarray], List[Sequence[Tuple[int, int, int]]], List[np.ndarray]
    ],
    test_data: Tuple[
        List[np.ndarray], List[Sequence[Tuple[int, int, int]]], List[np.ndarray]
    ],
    patch_shape: int,
    batch_size: int = 1,
    lr: float = 1e-4,
    logger=ClassificationLogger,
    trainer_class=ClassificationTrainer,
    n_iterations: int = int(1e5),
    check: bool = False,
    out_channels: int = 2,
    in_channels: int = 1,
    loss: Optional[torch.nn.Module] = None,
    metric: Optional[ClassificationMetric] = None,
    augmentations: Optional[Callable] = None,  # TODO: Replace with real augmentation pipeline
    normalization: Optional[Callable] = None,  # TODO: Replace with real normalization
    save_root: Optional[str] = None,
    n_samples_train: Optional[int] = None,
    n_samples_val: Optional[int] = None,
    dataset_class=ClassificationDataset,
    **kwargs,
):
    """Set up and run a classification training workflow.

    Args:
        name: Checkpoint name.
        train_data, val_data, test_data: Each is a 4-tuple of:
            - raw volumes: List[np.ndarray]
            - coordinates: List[Sequence[Tuple[int, int, int]]]
            - labels: List[np.ndarray]
            - label metadata: List[Sequence]
        patch_shape: Shape of input patch.
        batch_size: Batch size.
        lr: Learning rate.
        logger: Logger class.
        trainer_class: Trainer class.
        n_iterations: Number of training iterations.
        check: If True, checks data loaders and exits.
        out_channels: Number of output classes.
        in_channels: Number of input channels.
        loss: Loss function.
        metric: Evaluation metric.
        augmentations: Callable transformation for augmentation.
        normalization: Callable transformation for normalization.
        save_root: Directory to save checkpoints.
        n_samples_train: Optional limit for training samples.
        n_samples_val: Optional limit for validation samples.
        dataset_class: Dataset wrapper class.
        kwargs: Additional args for trainer.
    """
    num_workers = 6  #TODO using this in location training as well, check if it should be different

    #TODO actually get the test_loader and store the data from it as subtomograms somewhere, is this possible?
    train_loader, val_loader, test_loader = create_data_loader(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        normalization=normalization,
        augmentation=augmentations,
        patch_shape=patch_shape,
        num_workers=num_workers,
        batch_size=batch_size,
        dataset_class=dataset_class,
        n_samples_train=n_samples_train,
        n_samples_val=n_samples_val,
        n_classes=out_channels,
    )

    if check:
        from torch_em.util.debug import check_loader

        check_loader(train_loader, n_samples=4)
        check_loader(val_loader, n_samples=4)
        return

    model = get_3d_model(in_channels=in_channels, out_channels=out_channels)

    # Set the default loss and metric (if no values where passed).
    loss = torch.nn.CrossEntropyLoss() if loss is None else loss
    metric = ClassificationMetric() if metric is None else metric

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
        logger=logger,
        trainer_class=trainer_class,
        **kwargs,
    )

    trainer.fit(n_iterations)

