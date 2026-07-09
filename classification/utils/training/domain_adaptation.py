"""Unsupervised domain adaptation for the protein *classification* model, following the
Mean-Teacher approach used in SynapseNet, adapted for an image-level classifier.

Unlike detection (a dense output), the classifier consumes a subtomogram *crop* around
a candidate coordinate and predicts a single protein class. Mean-Teacher adaptation for
this classifier is a FixMatch-style recipe:

- the teacher (EMA of the student) predicts class probabilities on a **weakly**
  augmented crop; its ``argmax`` is the pseudo-label and it is kept only when the
  softmax confidence exceeds ``confidence_threshold``;
- the student is trained with cross-entropy against that pseudo-label on a **strongly**
  augmented view of the same crop.

Because the labels are image-level, no invertible / geometric bookkeeping is required:
the two augmented views can be produced fully independently.

The crop coordinates in the (unlabeled) target domain are produced by the caller (see
``classification_scripts/train_classification_domain_adaptation.py``) and passed in as a
per-tomogram list of ``(z, y, x)`` integer coordinates.
"""

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch_em
import torch_em.self_training as self_training
from torch_em.self_training.logger import SelfTrainingTensorboardLogger
from torch_em.classification.classification_logger import make_grid as classification_make_grid

from classification.training import get_single_subtomogram, get_volume
from classification.config import MAX_EXTENT_HALO
from .training import get_3d_model
from .cryoET_augmentation import CryoETAugment, random_rot90, random_flip


# ---------------------------------------------------------------------------
# Augmentations.
# ---------------------------------------------------------------------------
class WeakClassificationAugment:
    """Weak augmentation for the teacher view: orientation changes only.

    Proteins have no preferred orientation, so random 90-degree rotations and flips are
    label-preserving and do not distort the signal used to form a pseudo-label.
    """

    def __call__(self, volume):
        if not isinstance(volume, torch.Tensor):
            volume = torch.as_tensor(volume, dtype=torch.float32)
        volume = random_rot90(volume)
        volume = random_flip(volume)
        return volume


class UnsupervisedClassificationDataset(torch.utils.data.Dataset):
    """Yields two augmented views of an unlabeled subtomogram crop.

    Args:
        paths: Target-domain tomogram paths.
        coords_per_tomogram: For each path, a list of ``(z, y, x)`` candidate coordinates.
        max_extent: Crop size (without halo), matching classification training.
        in_channels: Number of input channels.
        normalization: Optional normalization callable applied before augmentation.
        weak_augmentation: Augmentation for the teacher view.
        strong_augmentation: Augmentation for the student view.
        n_samples: Optional fixed number of samples per epoch.
    """

    def __init__(
        self,
        paths: List[str],
        coords_per_tomogram: Sequence[Sequence[Tuple[int, int, int]]],
        max_extent: int = 39,
        in_channels: int = 1,
        normalization=None,
        weak_augmentation=None,
        strong_augmentation=None,
        n_samples: Optional[int] = None,
    ):
        assert len(paths) == len(coords_per_tomogram)
        self.max_extent = max_extent
        self.in_channels = in_channels
        self.normalization = normalization
        self.weak_augmentation = weak_augmentation or WeakClassificationAugment()
        self.strong_augmentation = strong_augmentation or CryoETAugment()

        # Match the bounding box used by extract_subtomograms so we only keep in-bounds crops.
        bbox_size = max_extent + MAX_EXTENT_HALO
        if bbox_size % 2 == 0:
            bbox_size += 1
        half = bbox_size // 2

        valid_paths, valid_coords = [], []
        for path, coords in zip(paths, coords_per_tomogram):
            volume = get_volume(path)
            D, H, W = volume.shape
            for c in coords:
                z, y, x = (int(round(v)) for v in c)
                if (z - half >= 0 and z + half < D and
                        y - half >= 0 and y + half < H and
                        x - half >= 0 and x + half < W):
                    valid_paths.append(path)
                    valid_coords.append((z, y, x))

        self.paths = valid_paths
        self.coords = valid_coords
        self._n_samples = n_samples
        if len(self.coords) == 0:
            raise ValueError("No in-bounds target-domain crops were found for adaptation.")
        print(f"Unsupervised classification dataset: {len(self.coords)} crops from {len(paths)} tomograms")

    def __len__(self):
        return self._n_samples if self._n_samples is not None else len(self.coords)

    def _load_crop(self, index):
        real_index = index % len(self.coords)
        path = self.paths[real_index]
        coord = self.coords[real_index]
        # A dummy target is passed; it is not used for unsupervised training.
        x, _ = get_single_subtomogram(path, coord, self.max_extent, self.in_channels, target="unlabeled")
        if self.normalization is not None:
            x = self.normalization(x)
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)
        return x

    def __getitem__(self, index):
        try:
            x = self._load_crop(index)
        except (IndexError, ValueError):
            import random
            return self[random.randint(0, len(self.coords) - 1)]
        view_teacher = self.weak_augmentation(x.clone())
        view_student = self.strong_augmentation(x.clone())
        return view_teacher, view_student


# ---------------------------------------------------------------------------
# FixMatch-style pseudo-labeler and losses for classification.
# ---------------------------------------------------------------------------
class ClassificationPseudoLabeler:
    """Turn teacher class logits into hard pseudo-labels + a per-sample confidence mask.

    Args:
        confidence_threshold: Keep a sample only if the teacher's max softmax probability
            is at least this value. ``None`` keeps every sample.
    """

    def __init__(self, confidence_threshold: Optional[float] = 0.9):
        self.confidence_threshold = confidence_threshold

    def __call__(self, teacher: nn.Module, input_: torch.Tensor):
        logits = teacher(input_)
        probs = torch.softmax(logits, dim=1)
        confidence, pseudo_labels = probs.max(dim=1)
        if self.confidence_threshold is None:
            label_filter = None
        else:
            label_filter = confidence >= self.confidence_threshold
        return pseudo_labels, label_filter

    def step(self, metric, epoch):
        pass


class ClassificationSelfTrainingLoss(nn.Module):
    """Cross-entropy between the student logits and the teacher pseudo-labels.

    Only samples accepted by the confidence mask contribute to the loss.
    """

    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction="none")
        self.init_kwargs = {}

    def _masked_ce(self, prediction, labels, label_filter):
        per_sample = self.ce(prediction, labels)
        if label_filter is None:
            return per_sample.mean()
        mask = label_filter.to(per_sample.dtype)
        denom = mask.sum().clamp_min(1.0)
        return (per_sample * mask).sum() / denom

    def __call__(self, model, input_, labels, label_filter=None):
        prediction = model(input_)
        return self._masked_ce(prediction, labels, label_filter)


class ClassificationSelfTrainingLossAndMetric(nn.Module):
    """Loss (as above) plus a validation metric (error rate = 1 - accuracy)."""

    def __init__(self):
        super().__init__()
        self.loss = ClassificationSelfTrainingLoss()
        self.init_kwargs = {}

    def __call__(self, model, input_, labels, label_filter=None):
        prediction = model(input_)
        loss = self.loss._masked_ce(prediction, labels, label_filter)
        pred_labels = prediction.argmax(dim=1)
        if label_filter is not None and label_filter.any():
            mask = label_filter.bool()
            accuracy = (pred_labels[mask] == labels[mask]).float().mean()
        else:
            accuracy = (pred_labels == labels).float().mean()
        # Lower is better, to match ReduceLROnPlateau(mode="min") and torch_em conventions.
        metric = 1.0 - accuracy
        return loss, metric


# ---------------------------------------------------------------------------
# Logging.
# ---------------------------------------------------------------------------
class ClassificationSelfTrainingLogger(SelfTrainingTensorboardLogger):
    """Tensorboard logger for classification adaptation.

    The default :class:`SelfTrainingTensorboardLogger` renders the prediction / pseudo-labels
    as dense (5D) image grids, which is wrong for a scalar classifier and crashes. We instead
    log the scalar losses / metric / learning rate / confidence threshold, plus crop-with-caption
    image tabs built with the same ``make_grid`` used by the non-DA ``ClassificationLogger``:
    each tile is the middle z-slice of a crop, captioned with the teacher pseudo-label (``t:``)
    and the student prediction (``p:``). Logs are written to ``<save_root>/logs/<name>``.
    """

    def _add_crop_images(self, step, name, crops, pseudo_labels, pred):
        """Log a grid of input crops captioned with pseudo-label (t:) and prediction (p:)."""
        # ``pred`` is only computed on image-log iterations during training; skip if absent.
        if pred is None:
            return
        grid = classification_make_grid(crops, target=pseudo_labels, prediction=pred)
        self.tb.add_image(tag=f"{name}/crops-pseudolabel-prediction", img_tensor=grid, global_step=step)

    def log_train_unsupervised(self, step, loss, x1, x2, pred, pseudo_labels, label_filter=None):
        self.tb.add_scalar(tag="train/unsupervised/loss", scalar_value=loss, global_step=step)
        if step % self.log_image_interval == 0:
            # x1 is the weak (teacher) view that produced the pseudo-label.
            self._add_crop_images(step, "train", x1, pseudo_labels, pred)

    def log_validation_unsupervised(self, step, metric, loss, x1, x2, pred, pseudo_labels, label_filter=None):
        self.tb.add_scalar(tag="validation/unsupervised/loss", scalar_value=loss, global_step=step)
        self.tb.add_scalar(tag="validation/unsupervised/metric", scalar_value=metric, global_step=step)
        self._add_crop_images(step, "validation", x1, pseudo_labels, pred)

    def log_train_supervised(self, step, loss, x, y, pred):
        self.tb.add_scalar(tag="train/supervised/loss", scalar_value=loss, global_step=step)
        if step % self.log_image_interval == 0:
            self._add_crop_images(step, "train_supervised", x, y, pred)

    def log_validation_supervised(self, step, metric, loss, x, y, pred):
        self.tb.add_scalar(tag="validation/supervised/loss", scalar_value=loss, global_step=step)
        self.tb.add_scalar(tag="validation/supervised/metric", scalar_value=metric, global_step=step)
        self._add_crop_images(step, "validation_supervised", x, y, pred)


# ---------------------------------------------------------------------------
# Model loading + adaptation entry point.
# ---------------------------------------------------------------------------
def _load_source_model(source_checkpoint: str, in_channels: int, out_channels: int,
                       use_efficientnet: bool = False) -> Tuple[nn.Module, dict]:
    """Load the classification model + source weights from a torch_em checkpoint dir.

    Returns the model and the source checkpoint's ``init`` dict (used to inherit
    ``patch_shape`` / ``idx_to_label`` for the adapted checkpoint).
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = get_3d_model(EfficientNet=use_efficientnet, in_channels=in_channels, out_channels=out_channels)
    ckpt_path = os.path.join(source_checkpoint, "best.pt") if os.path.isdir(source_checkpoint) else source_checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    source_init = checkpoint.get("init", {}) or {}
    return model.to(device), source_init


def _finalize_checkpoints(checkpoint_folder: str, patch_shape, idx_to_label: Optional[Dict]) -> None:
    """Make adapted checkpoints loadable by the standard classification inference.

    ``run_protein_classification.py`` reads ``checkpoint['init']['patch_shape']`` and
    ``checkpoint['init']['idx_to_label']``. The ``MeanTeacherTrainer`` does not know about
    these custom fields, so we inject them (inherited from the source model) into every
    checkpoint it wrote. We also mirror them at the top level, matching the non-DA
    checkpoints produced by ``ProteinClassificationTrainer``.
    """
    if idx_to_label is None or patch_shape is None:
        print("[warn] Could not inherit patch_shape/idx_to_label from the source checkpoint; "
              "the adapted checkpoint may not be directly usable for inference.")

    for name in ("best.pt", "latest.pt"):
        ckpt_path = os.path.join(checkpoint_folder, name)
        if not os.path.exists(ckpt_path):
            continue
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        init = checkpoint.get("init", {}) or {}
        init["patch_shape"] = patch_shape
        init["idx_to_label"] = idx_to_label
        checkpoint["init"] = init
        # Mirror at the top level too (parity with ProteinClassificationTrainer checkpoints).
        checkpoint["patch_shape"] = patch_shape
        checkpoint["idx_to_label"] = idx_to_label
        torch.save(checkpoint, ckpt_path)
        print(f"Patched {ckpt_path} with patch_shape={patch_shape} and "
              f"idx_to_label ({len(idx_to_label) if idx_to_label else 0} classes)")


def get_unsupervised_loader(
    paths: List[str],
    coords_per_tomogram: Sequence[Sequence[Tuple[int, int, int]]],
    max_extent: int,
    batch_size: int,
    in_channels: int = 1,
    normalization=None,
    n_samples: Optional[int] = None,
    num_workers: int = 8,
) -> torch.utils.data.DataLoader:
    """Build a dataloader yielding ``(teacher_view, student_view)`` crop pairs."""
    ds = UnsupervisedClassificationDataset(
        paths=paths,
        coords_per_tomogram=coords_per_tomogram,
        max_extent=max_extent,
        in_channels=in_channels,
        normalization=normalization,
        n_samples=n_samples,
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=False,
    )
    loader.shuffle = True
    return loader


def mean_teacher_adaptation(
    name: str,
    source_checkpoint: str,
    train_paths: List[str],
    val_paths: List[str],
    train_coords: Sequence[Sequence[Tuple[int, int, int]]],
    val_coords: Sequence[Sequence[Tuple[int, int, int]]],
    max_extent: int,
    n_classes: int,
    in_channels: int = 1,
    normalization=None,
    confidence_threshold: float = 0.9,
    batch_size: int = 32,
    lr: float = 1e-4,
    n_iterations: int = int(2e3),
    n_samples_train: Optional[int] = None,
    n_samples_val: Optional[int] = None,
    momentum: float = 0.999,
    use_efficientnet: bool = False,
    save_root: Optional[str] = None,
    num_workers: int = 8,
    check: bool = False,
    patch_shape: Optional[Tuple[int, int, int]] = None,
    idx_to_label: Optional[Dict[int, str]] = None,
) -> None:
    """Run Mean-Teacher domain adaptation for the protein classification model.

    Args:
        name: Name of the adapted-model checkpoint.
        source_checkpoint: Path to the trained source classification model (torch_em
            checkpoint directory containing ``best.pt``). Its class-index ordering is
            inherited, so the pseudo-labels are consistent with the source model.
        train_paths / val_paths: Target-domain tomograms for train / val.
        train_coords / val_coords: Per-tomogram candidate ``(z, y, x)`` coordinates.
        max_extent: Crop size (without halo), matching classification training.
        n_classes: Number of classes (must match the source model's output channels).
        in_channels: Number of input channels.
        normalization: Normalization callable (e.g. ``CryoETNormalize``).
        confidence_threshold: Minimum teacher softmax confidence to keep a pseudo-label.
        batch_size: Batch size.
        lr: Initial learning rate.
        n_iterations: Number of training iterations.
        n_samples_train / n_samples_val: Optional samples per epoch.
        momentum: EMA momentum for the teacher weights.
        use_efficientnet: Whether the source model is an EfficientNet3D (else ResNet3D-18).
        save_root: Root folder for the checkpoint and logs.
        num_workers: Dataloader workers.
        check: If True, only visualize the loaders instead of training.
    """
    model, source_init = _load_source_model(source_checkpoint, in_channels, n_classes, use_efficientnet)
    print("Mean-Teacher classification adaptation initialized from:", source_checkpoint)

    # Inherit the crop size / class mapping from the source model so the adapted
    # checkpoint is loadable by the standard classification inference. Prefer explicit
    # overrides, but fall back to the source checkpoint's own values (which carry the
    # correct integer class keys and (D, H, W) patch shape).
    patch_shape = patch_shape if patch_shape is not None else source_init.get("patch_shape")
    idx_to_label = idx_to_label if idx_to_label is not None else source_init.get("idx_to_label")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    pseudo_labeler = ClassificationPseudoLabeler(confidence_threshold=confidence_threshold)
    loss = ClassificationSelfTrainingLoss()
    loss_and_metric = ClassificationSelfTrainingLossAndMetric()

    unsupervised_train_loader = get_unsupervised_loader(
        train_paths, train_coords, max_extent, batch_size, in_channels,
        normalization=normalization, n_samples=n_samples_train, num_workers=num_workers,
    )
    unsupervised_val_loader = get_unsupervised_loader(
        val_paths, val_coords, max_extent, batch_size, in_channels,
        normalization=normalization, n_samples=n_samples_val, num_workers=num_workers,
    )

    if check:
        from torch_em.util.debug import check_loader
        check_loader(unsupervised_train_loader, n_samples=4)
        check_loader(unsupervised_val_loader, n_samples=4)
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
        unsupervised_train_loader=unsupervised_train_loader,
        unsupervised_val_loader=unsupervised_val_loader,
        supervised_train_loader=None,
        supervised_val_loader=None,
        # Scalar-only logger: the default self-training logger logs 5D image grids,
        # which do not apply to a scalar classifier. Writes to <save_root>/logs/<name>.
        logger=ClassificationSelfTrainingLogger,
        mixed_precision=True,
        compile_model=False,
        device=device,
        reinit_teacher=False,
        momentum=momentum,
        save_root=save_root,
    )
    trainer.fit(n_iterations)

    # Inject patch_shape / idx_to_label into the saved checkpoints so the adapted model
    # can be run with the same inference script as the non-DA classification models.
    _finalize_checkpoints(trainer.checkpoint_folder, patch_shape, idx_to_label)
