from typing import Any, Dict, Optional, Callable, Tuple, Union

import torch
from torch.utils.data import DataLoader
from torch_em.classification.classification_trainer import ClassificationTrainer

from torch_em.trainer.tensorboard_logger import TensorboardLogger
from torch_em.loss import DiceLoss

DEFAULT_SCHEDULER_KWARGS = {"mode": "min", "factor": 0.5, "patience": 5}


class ProteinClassificationTrainer(ClassificationTrainer):
    def __init__(self, *args, patch_shape=None, idx_to_label=None, **kwargs):
        self.args = args
        self._kwargs = kwargs
        # Remove the custom parameters from kwargs before passing to parent
        kwargs.pop('patch_shape', None)
        kwargs.pop('idx_to_label', None)

        super().__init__(*args, **kwargs)
        self.patch_shape = patch_shape
        self.idx_to_label = idx_to_label

    def save_checkpoint(self, name, current_metric, best_metric, train_time=0.0, **extra_save_dict):
        """Override the save_checkpoint method to include custom parameters."""
        # Call the parent class's save_checkpoint method with additional parameters
        super().save_checkpoint(
            name,
            current_metric,
            best_metric,
            train_time=train_time,
            patch_shape=self.patch_shape,
            idx_to_label=self.idx_to_label,
            **extra_save_dict
        )

    class Serializer(ClassificationTrainer.Serializer):
        def dump_args(self, kwarg_name: str):
            assert kwarg_name == "args"
            self.init_data[kwarg_name] = self.trainer.args

        """Override the Serializer to handle custom parameters."""
        def dump_patch_shape(self, kwarg_name: str):
            assert kwarg_name == "patch_shape"
            self.init_data[kwarg_name] = self.trainer.patch_shape

        def dump_idx_to_label(self, kwarg_name: str):
            assert kwarg_name == "idx_to_label"
            self.init_data[kwarg_name] = self.trainer.idx_to_label

    class Deserializer(ClassificationTrainer.Deserializer):
        def load_args(self, kwarg_name: str, optional: bool):
            assert kwarg_name == "args"
            self.trainer_kwargs[kwarg_name] = self.init_data[kwarg_name]

        """Override the Deserializer to handle custom parameters."""
        def load_patch_shape(self, kwarg_name: str, optional: bool):
            assert kwarg_name == "patch_shape"
            self.trainer_kwargs[kwarg_name] = self.init_data[kwarg_name]

        def load_idx_to_label(self, kwarg_name: str, optional: bool):
            assert kwarg_name == "idx_to_label"
            self.trainer_kwargs[kwarg_name] = self.init_data[kwarg_name]


def default_classification_trainer(
    name: str,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss: Optional[torch.nn.Module] = None,
    metric: Optional[Callable] = None,
    learning_rate: float = 1e-3,
    device: Optional[Union[str, torch.device]] = None,
    log_image_interval: int = 100,
    mixed_precision: bool = True,
    early_stopping: Optional[int] = None,
    logger=TensorboardLogger,
    logger_kwargs: Optional[Dict[str, Any]] = None,
    scheduler_kwargs: Dict[str, Any] = DEFAULT_SCHEDULER_KWARGS,
    optimizer_kwargs: Dict[str, Any] = {},
    trainer_class=ClassificationTrainer,
    id_: Optional[str] = None,
    save_root: Optional[str] = None,
    compile_model: Optional[Union[bool, str]] = None,
    rank: Optional[int] = None,
    patch_shape: Optional[Tuple[int, int, int]] = None,
    idx_to_label: Optional[Dict[int, str]] = None,
    encoder_lr_scale: float = 1.0,
):
    """Get a trainer for a classification network.


    Args:
        name: The name of the checkpoint that will be created by the trainer.
        model: The model to train.
        train_loader: The data loader containing the training data.
        val_loader: The data loader containing the validation data.
        loss: The loss function for training.
        metric: The metric for validation.
        learning_rate: The initial learning rate for the AdamW optimizer.
        device: The torch device to use for training. If None, will use a GPU if available.
        log_image_interval: The interval for saving images during logging, in training iterations.
        mixed_precision: Whether to train with mixed precision.
        early_stopping: The patience for early stopping in epochs. If None, early stopping will not be used.
        logger: The logger class. Will be instantiated for logging.
            By default uses `torch_em.training.tensorboard_logger.TensorboardLogger`.
        logger_kwargs: The keyword arguments for the logger class.
        scheduler_kwargs: The keyword arguments for ReduceLROnPlateau.
        optimizer_kwargs: The keyword arguments for the AdamW optimizer.
        trainer_class: The trainer class. Uses `torch_em.classification.classification_trainer.ClassificationTrainer` by default,
            but can be set to a custom trainer class to enable custom training procedures.
        id_: Unique identifier for the trainer. If None then `name` will be used.
        save_root: The root folder for saving the checkpoint and logs.
        compile_model: Whether to compile the model before training.
        rank: Rank argument for distributed training. See `torch_em.multi_gpu_training` for details.
        encoder_lr_scale: Learning-rate multiplier for the backbone (everything but `fc`).
            Use < 1 when fine-tuning a contrastively pretrained encoder.

    Returns:
        The trainer.
    """
    if encoder_lr_scale != 1.0:
        head = [p for n, p in model.named_parameters() if n.startswith("fc.")]
        backbone = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
        params = [{"params": backbone, "lr": learning_rate * encoder_lr_scale}, {"params": head, "lr": learning_rate}]
        print(f"Fine-tuning with backbone lr={learning_rate * encoder_lr_scale:g}, head lr={learning_rate:g}")
    else:
        params = model.parameters()
    optimizer = torch.optim.AdamW(params, lr=learning_rate, **optimizer_kwargs)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_kwargs)

    loss = DiceLoss() if loss is None else loss
    metric = DiceLoss() if metric is None else metric

    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device(device)

    # CPU does not support mixed precision training.
    if device.type == "cpu":
        mixed_precision = False

    return trainer_class(
        name=name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss=loss,
        metric=metric,
        optimizer=optimizer,
        device=device,
        lr_scheduler=scheduler,
        mixed_precision=mixed_precision,
        early_stopping=early_stopping,
        log_image_interval=log_image_interval,
        logger=logger,
        logger_kwargs=logger_kwargs,
        id_=id_,
        save_root=save_root,
        compile_model=compile_model,
        rank=rank,
        patch_shape=patch_shape,
        idx_to_label=idx_to_label
    )

'''def default_classification_trainer(
    name: str,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss: Optional[torch.nn.Module] = None,
    metric: Optional[Callable] = None,
    learning_rate: float = 1e-3,
    device: Optional[Union[str, torch.device]] = None,
    log_image_interval: int = 100,
    mixed_precision: bool = True,
    early_stopping: Optional[int] = None,
    logger=TensorboardLogger,
    logger_kwargs: Optional[Dict[str, Any]] = None,
    scheduler_kwargs: Dict[str, Any] = DEFAULT_SCHEDULER_KWARGS,
    optimizer_kwargs: Dict[str, Any] = {},
    trainer_class=ClassificationTrainer,
    id_: Optional[str] = None,
    save_root: Optional[str] = None,
    compile_model: Optional[Union[bool, str]] = None,
    rank: Optional[int] = None,
    patch_shape: Optional[int] = None,
    idx_to_label: Optional[Dict[int, str]] = None,
):
    """Get a trainer for a classification network."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, **optimizer_kwargs)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_kwargs)

    loss = DiceLoss() if loss is None else loss
    metric = DiceLoss() if metric is None else metric

    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device(device)

    # CPU does not support mixed precision training.
    if device.type == "cpu":
        mixed_precision = False

    # Prepare the kwargs for the trainer
    trainer_kwargs = {
        "name": name,
        "model": model,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "loss": loss,
        "metric": metric,
        "optimizer": optimizer,
        "device": device,
        "lr_scheduler": scheduler,
        "mixed_precision": mixed_precision,
        "early_stopping": early_stopping,
        "log_image_interval": log_image_interval,
        "logger": logger,
        "logger_kwargs": logger_kwargs,
        "id_": id_,
        "save_root": save_root,
        "compile_model": compile_model,
        "rank": rank,
    }

    # Add custom parameters if the trainer class is ClassificationTrainer
    if trainer_class == ClassificationTrainer:
        print("here")
        trainer_kwargs["patch_shape"] = patch_shape
        trainer_kwargs["idx_to_label"] = idx_to_label

    return trainer_class(**trainer_kwargs)'''