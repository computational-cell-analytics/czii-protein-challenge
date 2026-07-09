"""Unsupervised (and optionally semi-supervised) domain adaptation for the protein
*detection* model, following the Mean-Teacher approach used in SynapseNet.

The detection network is an ``AnisotropicUNet`` that predicts a dense output with
``out_channels = 5`` (1 detection heatmap + 4 stereographic-flow channels) and is
trained with :class:`detection.utils.training.training.CombinedLoss` (MSE on the
heatmap + weighted MSE on the flow).

Mean-Teacher adaptation transfers a model trained on a *source* domain to a *target*
domain **without requiring labels in the target domain**. A teacher model (an
exponential moving average of the student weights) predicts pseudo-labels on the
unlabeled target data, and the student is trained to reproduce them under a different
augmentation.

Design note on augmentations
----------------------------
SynapseNet uses ``MeanTeacherTrainerWithInvertibleAugmentations``, which applies
*geometric* augmentations (flips / rotations) and inverts them on the prediction
before computing the consistency loss. That is correct for a scalar segmentation
probability, but it is **not** correct for our flow channels: the flow is a vector
field, so a flip/rotation would need to also flip/rotate the vector *components*, not
merely reposition the voxels. Inverting the spatial transform therefore leaves the
flow channels inconsistent.

To keep the flow consistent we use the plain ``MeanTeacherTrainer`` together with
``torch_em.data.RawDataset`` configured to return two independently **intensity-only**
augmented views of the same patch (Gaussian noise / blur / intensity scaling). No
geometric transform is applied, so both the heatmap and the flow stay spatially and
directionally aligned between the teacher pseudo-labels and the student input.
Intensity perturbations are also precisely the kind of domain shift (noise / contrast)
that we want the detection model to become robust to in cryo-ET data.
"""

import os
import random
from typing import Optional, Tuple

import torch
import torch_em
import torch_em.self_training as self_training
from torch_em.data import RawDataset
from torch_em.data.concat_dataset import ConcatDataset

from .detection_dataset import DetectionDataset, robust_standardize
from .data_loader import create_data_loader
from .training import get_3d_model, CombinedLoss
from ..transform import HeatmapFlowTransform

from detection.config import FLOW_SIGMA


# ---------------------------------------------------------------------------
# Intensity-only weak augmentations (operate on a (C, D, H, W) float tensor).
# Written as a small picklable class so the RawDataset can be serialized with
# the trainer checkpoint.
# ---------------------------------------------------------------------------
class WeakIntensityAugmentation:
    """Random intensity perturbations that do **not** move voxels around.

    Applied to an already standardized ``(C, D, H, W)`` tensor. Because no
    geometric transform is used, the detection heatmap and the flow vectors
    remain valid after augmentation.
    """

    def __init__(
        self,
        noise_prob: float = 0.75,
        blur_prob: float = 0.5,
        intensity_prob: float = 0.5,
        noise_std_range: Tuple[float, float] = (0.05, 0.2),
        blur_sigma_range: Tuple[float, float] = (0.3, 1.0),
        intensity_scale_range: Tuple[float, float] = (0.9, 1.1),
    ):
        self.noise_prob = noise_prob
        self.blur_prob = blur_prob
        self.intensity_prob = intensity_prob
        self.noise_std_range = noise_std_range
        self.blur_sigma_range = blur_sigma_range
        self.intensity_scale_range = intensity_scale_range

    def _add_noise(self, volume):
        std = random.uniform(*self.noise_std_range)
        return volume + torch.randn_like(volume) * std

    def _blur(self, volume):
        import math
        import torch.nn.functional as F

        sigma = random.uniform(*self.blur_sigma_range)
        size = int(2 * math.ceil(2 * sigma) + 1)
        coords = torch.arange(size, device=volume.device) - size // 2
        grid = torch.stack(torch.meshgrid(coords, coords, coords, indexing="ij"), dim=-1).float()
        kernel = torch.exp(-grid.pow(2).sum(-1) / (2 * sigma ** 2))
        kernel = (kernel / kernel.sum()).view(1, 1, size, size, size)
        # volume: (C, D, H, W) -> conv each channel independently
        c = volume.shape[0]
        vol = volume.unsqueeze(0)  # (1, C, D, H, W)
        kernel = kernel.expand(c, 1, size, size, size)
        blurred = F.conv3d(vol, kernel, padding=size // 2, groups=c)
        return blurred.squeeze(0)

    def _intensity(self, volume):
        scale = random.uniform(*self.intensity_scale_range)
        shift = torch.randn(1, device=volume.device) * volume.std() * random.uniform(0.0, 0.1)
        return volume * scale + shift

    def __call__(self, volume):
        if not isinstance(volume, torch.Tensor):
            volume = torch.as_tensor(volume, dtype=torch.float32)
        volume = volume.clone()
        if random.random() < self.intensity_prob:
            volume = self._intensity(volume)
        if random.random() < self.blur_prob:
            volume = self._blur(volume)
        if random.random() < self.noise_prob:
            volume = self._add_noise(volume)
        return volume


def get_unsupervised_loader(
    data_paths: Tuple[str],
    raw_key: Optional[str],
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    n_samples: Optional[int],
    augmentation: Optional[callable] = None,
    num_workers: Optional[int] = None,
) -> torch.utils.data.DataLoader:
    """Build a dataloader that yields two intensity-augmented views per patch.

    Args:
        data_paths: Filepaths to the (unlabeled) target-domain tomograms.
        raw_key: Internal key for the raw data (e.g. ``"0"`` for a multiscale ``.zarr``,
            ``None`` to let the loader auto-derive it for single-file formats).
        patch_shape: The training patch shape, e.g. ``(128, 256, 256)``.
        batch_size: The batch size.
        n_samples: Total number of samples per epoch (spread across the tomograms).
        augmentation: The weak augmentation applied to build the two views. Defaults to
            :class:`WeakIntensityAugmentation`.
        num_workers: Number of dataloader workers.

    Returns:
        A PyTorch dataloader whose items are ``(view1, view2)`` tensors.
    """
    if augmentation is None:
        augmentation = WeakIntensityAugmentation()
    augmentations = (augmentation, augmentation)

    if n_samples is None:
        n_samples_per_ds = None
    else:
        n_samples_per_ds = int(n_samples / len(data_paths))

    datasets = [
        RawDataset(
            raw_path=path,
            raw_key=raw_key,
            patch_shape=patch_shape,
            raw_transform=robust_standardize,
            transform=None,
            ndim=3,
            n_samples=n_samples_per_ds,
            with_channels=False,
            augmentations=augmentations,
        )
        for path in data_paths
    ]
    ds = ConcatDataset(*datasets)

    if num_workers is None:
        num_workers = 4 * batch_size
    loader = torch_em.segmentation.get_data_loader(
        ds, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    return loader


def _load_source_model(source_checkpoint: str) -> torch.nn.Module:
    """Load the detection model from a torch_em checkpoint dir or a serialized model."""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if os.path.isdir(source_checkpoint):
        # The checkpoint was saved with a dataset that imports this package as ``utils``.
        # Mirror the shim used in detection.utils.prediction so deserialization works.
        import sys
        import detection.utils as det_utils
        sys.modules.setdefault("utils", det_utils)
        model = torch_em.util.load_model(checkpoint=source_checkpoint, device=device)
    else:
        model = torch.load(source_checkpoint, weights_only=False)
    return model


def mean_teacher_adaptation(
    name: str,
    unsupervised_train_paths: Tuple[str],
    unsupervised_val_paths: Tuple[str],
    patch_shape: Tuple[int, int, int],
    source_checkpoint: Optional[str] = None,
    save_root: Optional[str] = None,
    supervised_train_paths: Optional[Tuple[str]] = None,
    supervised_val_paths: Optional[Tuple[str]] = None,
    supervised_train_label_paths: Optional[Tuple[str]] = None,
    supervised_val_label_paths: Optional[Tuple[str]] = None,
    confidence_threshold: Optional[float] = None,
    raw_key: Optional[str] = None,
    out_channels: int = 5,
    batch_size: int = 1,
    lr: float = 1e-4,
    n_iterations: int = int(1e4),
    n_samples_train: Optional[int] = None,
    n_samples_val: Optional[int] = None,
    momentum: float = 0.999,
    check: bool = False,
) -> None:
    """Run Mean-Teacher domain adaptation for the protein detection model.

    Two modes are supported (mirroring SynapseNet):
    - **unsupervised** (default): only the unlabeled target-domain data is used; the
      teacher is initialized from ``source_checkpoint``.
    - **semi-supervised**: additionally feed labeled source-domain data by passing the
      ``supervised_*_paths``; this anchors the model to the original task.

    Args:
        name: Name of the checkpoint to be trained.
        unsupervised_train_paths: Target-domain training tomograms (no labels needed).
        unsupervised_val_paths: Target-domain validation tomograms (no labels needed).
        patch_shape: Training patch shape, e.g. ``(128, 256, 256)``.
        source_checkpoint: Path to the trained source detection model (torch_em
            checkpoint directory or serialized model). Used to initialize the teacher
            and student. Required unless supervised training data is given for training
            from scratch.
        save_root: Root folder for the checkpoint and logs.
        supervised_train_paths / supervised_val_paths: Optional source-domain raw
            tomograms for semi-supervised adaptation.
        supervised_train_label_paths / supervised_val_label_paths: Labels matching the
            supervised raw tomograms.
        confidence_threshold: Optional threshold for masking the pseudo-labels. ``None``
            (default) computes the consistency loss on the full dense prediction, which
            is the appropriate choice for the regression heatmap + flow output.
        raw_key: Internal key for the raw data (``"0"`` for multiscale zarr, ``None`` for
            single-file formats).
        out_channels: Number of output channels of the detection U-Net (heatmap + flow).
        batch_size: Batch size for training.
        lr: Initial learning rate.
        n_iterations: Number of training iterations.
        n_samples_train / n_samples_val: Samples per epoch for train / val.
        momentum: EMA momentum for the teacher weights.
        check: If True, only visualize the loaders instead of training.
    """
    assert (supervised_train_paths is None) == (supervised_val_paths is None)
    semisupervised = supervised_train_paths is not None

    if source_checkpoint is None:
        assert semisupervised, "Training from scratch requires supervised training data."
        print("Mean-Teacher detection training from scratch.")
        model = get_3d_model(in_channels=1, out_channels=out_channels)
        reinit_teacher = True
    else:
        print("Mean-Teacher detection adaptation initialized from:", source_checkpoint)
        model = _load_source_model(source_checkpoint)
        reinit_teacher = False

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    # Self-training components. The detection output is a regression (heatmap + flow),
    # so we use the CombinedLoss as the internal consistency loss and (by default) do
    # not threshold the pseudo-labels.
    combined_loss = CombinedLoss(heatmap_weight=1.0, flow_weight=0.1)
    pseudo_labeler = self_training.DefaultPseudoLabeler(confidence_threshold=confidence_threshold)
    loss = self_training.DefaultSelfTrainingLoss(loss=combined_loss)
    loss_and_metric = self_training.DefaultSelfTrainingLossAndMetric(
        loss=combined_loss, metric=torch.nn.MSELoss(reduction="mean")
    )

    unsupervised_train_loader = get_unsupervised_loader(
        data_paths=unsupervised_train_paths, raw_key=raw_key, patch_shape=patch_shape,
        batch_size=batch_size, n_samples=n_samples_train,
    )
    unsupervised_val_loader = get_unsupervised_loader(
        data_paths=unsupervised_val_paths, raw_key=raw_key, patch_shape=patch_shape,
        batch_size=batch_size, n_samples=n_samples_val,
    )

    supervised_train_loader = None
    supervised_val_loader = None
    if semisupervised:
        assert supervised_train_label_paths is not None and supervised_val_label_paths is not None, \
            "Semi-supervised adaptation requires label paths for the supervised data."
        label_transform = HeatmapFlowTransform(
            eps=1e-5, sigma=None, lower_bound=None, upper_bound=None, flow_sigma=FLOW_SIGMA,
        )
        supervised_train_loader, supervised_val_loader, _ = create_data_loader(
            supervised_train_paths, supervised_train_label_paths,
            supervised_val_paths, supervised_val_label_paths,
            None, None,
            raw_transform=None, label_transform=label_transform, transform=None,
            patch_shape=patch_shape, num_workers=4 * batch_size, batch_size=batch_size,
            raw_key=raw_key, dataset_class=DetectionDataset,
            n_samples_train=n_samples_train, n_samples_val=n_samples_val,
        )

    if check:
        from torch_em.util.debug import check_loader
        check_loader(unsupervised_train_loader, n_samples=4)
        check_loader(unsupervised_val_loader, n_samples=4)
        if supervised_train_loader is not None:
            check_loader(supervised_train_loader, n_samples=4)
            check_loader(supervised_val_loader, n_samples=4)
        return

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    trainer = self_training.MeanTeacherTrainer(
        name=name,
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        pseudo_labeler=pseudo_labeler,
        unsupervised_loss=loss,
        unsupervised_loss_and_metric=loss_and_metric,
        supervised_train_loader=supervised_train_loader,
        unsupervised_train_loader=unsupervised_train_loader,
        supervised_val_loader=supervised_val_loader,
        unsupervised_val_loader=unsupervised_val_loader,
        supervised_loss=loss if semisupervised else None,
        supervised_loss_and_metric=loss_and_metric if semisupervised else None,
        logger=self_training.SelfTrainingTensorboardLogger,
        mixed_precision=True,
        log_image_interval=100,
        compile_model=False,
        device=device,
        reinit_teacher=reinit_teacher,
        momentum=momentum,
        save_root=save_root,
    )
    trainer.fit(n_iterations)
