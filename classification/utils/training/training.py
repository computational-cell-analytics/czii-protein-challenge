from typing import Callable, List, Optional, Tuple

import numpy as np
import sklearn.metrics as metrics
import torch
import torch_em

from torch_em.classification.classification_logger import ClassificationLogger
from .trainer import ProteinClassificationTrainer

from .data_loader import create_data_loader
from .classification_dataset import ClassificationDataset
from .trainer import default_classification_trainer
from .loss import FocalLossWithLabelSmoothing, BalancedSoftmaxLoss

from torch_em.model.resnet3d import resnet3d_18
import torch.nn as nn


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
    EfficientNet: True,
    in_channels: int,
    out_channels: int,
) -> torch.nn.Module:
    """Get the 3D model. ResNet or EfficientNet

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels (i.e., number of classes).
    """

    if EfficientNet:
        from external.efficientnet3d.efficientnet_pytorch_3d import EfficientNet3D
        model = EfficientNet3D.from_name("efficientnet-b0", override_params={'num_classes': out_channels}, in_channels=in_channels)
    else:
        
        model = resnet3d_18(in_channels=in_channels, out_channels=out_channels)
        
        # Replace conv1 and maxpool
        model.conv1 = nn.Conv3d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()  # remove pooling to preserve resolution

    return model


def classification_training(
    name: str,
    train_paths: List[str],
    val_paths: List[str],
    test_paths: List[str],
    max_extent: int,
    target_root: str,
    patch_shape: Tuple[int, int, int],
    batch_size: int = 1,
    lr: float = 1e-4,
    logger=ClassificationLogger,
    trainer_class=ProteinClassificationTrainer,
    n_iterations: int = int(1e5),
    check: bool = False,
    out_channels: int = 2,
    in_channels: int = 1,
    loss: Optional[torch.nn.Module] = None,
    metric: Optional[ClassificationMetric] = None,
    augmentations: Optional[Callable] = None,
    normalization: Optional[Callable] = None,
    save_root: Optional[str] = None,
    n_samples_train: Optional[int] = None,
    n_samples_val: Optional[int] = None,
    dataset_class=ClassificationDataset,
    pretrained_encoder: Optional[str] = None,
    pretrained_use_teacher: bool = False,
    encoder_lr_scale: float = 1.0,
    **kwargs,
):
    """Set up and run a classification training workflow.

    Args:
        name: Checkpoint name.
        train_data, val_data, test_data: Each is a List of paths to the full tomograms
        max_extent: size of bbox for the subtomograms of the proteins
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
        pretrained_encoder: Checkpoint of a contrastively pretrained encoder to initialize
            the ResNet backbone from (see `classification.utils.training.contrastive`).
        pretrained_use_teacher: Take the EMA teacher weights from that checkpoint.
        encoder_lr_scale: Learning-rate multiplier for the pretrained backbone relative to the
            (randomly initialized) classification head. Below 1 preserves the pretrained
            features during fine-tuning.
        kwargs: Additional args for trainer.
    """
    
    num_workers = kwargs.pop("num_workers", 4 * batch_size)

    train_loader, val_loader, test_loader, idx_to_label = create_data_loader(
        train_data=train_paths,
        val_data=val_paths,
        test_data=test_paths,
        in_channels=in_channels,
        max_extent=max_extent,
        target_root=target_root,
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

    model = get_3d_model(EfficientNet=False, in_channels=in_channels, out_channels=out_channels)

    if pretrained_encoder is not None:
        from .contrastive import load_pretrained_backbone
        model = load_pretrained_backbone(model, pretrained_encoder, prefer_teacher=pretrained_use_teacher)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"\nModel on device: {device}")

    # Set the default loss and metric (if no values where passed).
    loss = torch.nn.CrossEntropyLoss() if loss is None else loss
    metric = ClassificationMetric() if metric is None else metric

    # If the focal loss is set to data-balanced alpha, compute inverse-frequency
    # class weights from the training targets. Class counts are derived at runtime,
    # so they never need to be hard-coded per dataset.
    if isinstance(loss, FocalLossWithLabelSmoothing) and loss.needs_alpha:
        train_set = train_loader.dataset
        counts, alpha = loss.set_alpha_from_targets(
            train_set.targets, label_to_index=train_set.label_to_index
        )
        print(f"[FocalLoss] train class counts (by index): {counts.tolist()}")
        print(f"[FocalLoss] inverse-frequency alpha:       {alpha.tolist()}")

    # Balanced Softmax needs the training class frequencies as its log-prior.
    if isinstance(loss, BalancedSoftmaxLoss) and loss.needs_prior:
        train_set = train_loader.dataset
        counts, prior = loss.set_prior_from_targets(
            train_set.targets, label_to_index=train_set.label_to_index
        )
        print(f"[BalancedSoftmax] train class counts (by index): {counts.tolist()}")
        print(f"[BalancedSoftmax] class-frequency prior:         {prior.tolist()}")

    trainer = default_classification_trainer(
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
        patch_shape=patch_shape,
        idx_to_label=idx_to_label,
        encoder_lr_scale=encoder_lr_scale,
        **kwargs,
    )
    
    trainer.fit(n_iterations)

    return idx_to_label
