"""MoCo-style contrastive pretraining of the ResNet3d subtomogram encoder (stage 1).

Two physically plausible corruptions of the same crop are pulled together in embedding space.
An EMA teacher produces the targets -- a much more stable signal than the online encoder at
cryo-ET SNRs. Stage 2 loads the backbone into the classifier via
``classification_training(..., pretrained_encoder=...)``.
"""

import os
import time
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_em

from .contrastive_dataset import ContrastivePairDataset
from .noise_augmentation import NoiseAwareViewAugment, snr_positive_weight
from .training import get_3d_model


# ---------------------------------------------------------------------------
# Model.
# ---------------------------------------------------------------------------
def _mlp(in_dim: int, hidden_dim: int, out_dim: int, last_bn: bool = False) -> nn.Sequential:
    layers = [nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True),
              nn.Linear(hidden_dim, out_dim)]
    if last_bn:
        layers.append(nn.BatchNorm1d(out_dim, affine=False))
    return nn.Sequential(*layers)


class ContrastiveEncoder(nn.Module):
    """ResNet3d backbone + projection head (+ prediction head for the online branch).

    The backbone comes from the same :func:`get_3d_model` as supervised training with ``fc``
    replaced by an identity, so its state_dict transfers key-for-key into the classifier.

    Args:
        use_predictor: Asymmetric prediction head (MoCo v3 / BYOL); reduces the representation
            gap between the online encoder and the EMA teacher.
    """

    feature_dim = 512  # resnet3d_18 output width

    def __init__(
        self,
        in_channels: int = 1,
        projection_dim: int = 128,
        hidden_dim: int = 1024,
        use_predictor: bool = True,
    ):
        super().__init__()
        self.init_kwargs = {
            "in_channels": in_channels, "projection_dim": projection_dim,
            "hidden_dim": hidden_dim, "use_predictor": use_predictor,
        }
        self.backbone = get_3d_model(EfficientNet=False, in_channels=in_channels, out_channels=1)
        self.backbone.fc = nn.Identity()
        self.projector = _mlp(self.feature_dim, hidden_dim, projection_dim, last_bn=True)
        self.predictor = _mlp(projection_dim, hidden_dim, projection_dim) if use_predictor else None

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor, predict: bool = False) -> torch.Tensor:
        z = self.projector(self.backbone(x))
        if predict and self.predictor is not None:
            z = self.predictor(z)
        return F.normalize(z.float(), dim=1)


# ---------------------------------------------------------------------------
# Loss.
# ---------------------------------------------------------------------------
class NoiseAwareInfoNCE(nn.Module):
    """InfoNCE over positive pairs, with a selectable rule for what counts as a negative.

    Three modes, differing only in how heavily-noised or same-class samples are treated:

    ``downweight``
        Ours. Negatives are other crops. A pair whose worse view falls below
        ``zero_weight_db`` is dropped from the positive loss but never pushed apart.
    ``nrcl``
        CryoEngine's NRCL (arXiv:2509.24311). No down-weighting; instead an extra
        noise-aware InfoNCE term treats an information-destroying view of the *same*
        crop as an explicit negative. Total = L_instance + L_noise, as in the paper.
    ``supervised``
        SupCon-style. Labels decide: every same-class key is a positive, and negatives
        are drawn only from other classes.

    Args:
        full_weight_db / zero_weight_db: SNR band over which the pair weight ramps 1 -> 0
            (``downweight`` only).
        nrcl_weight: Weight on the noise-aware term (``nrcl`` only).
    """

    MODES = ("downweight", "nrcl", "supervised")

    def __init__(
        self,
        temperature: float = 0.2,
        full_weight_db: float = 3.0,
        zero_weight_db: float = -6.0,
        mode: str = "downweight",
        nrcl_weight: float = 1.0,
    ):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"Unknown mode '{mode}'. Choose from {self.MODES}.")
        self.temperature = temperature
        self.full_weight_db = full_weight_db
        self.zero_weight_db = zero_weight_db
        self.mode = mode
        self.nrcl_weight = nrcl_weight
        self.init_kwargs = {
            "temperature": temperature, "full_weight_db": full_weight_db,
            "zero_weight_db": zero_weight_db, "mode": mode, "nrcl_weight": nrcl_weight,
        }

    @property
    def needs_noisy_view(self) -> bool:
        return self.mode == "nrcl"

    @property
    def needs_labels(self) -> bool:
        return self.mode == "supervised"

    def pair_weights(self, snr1: torch.Tensor, snr2: torch.Tensor) -> torch.Tensor:
        """Weight from the worse of the two views. Uniform unless mode is ``downweight``."""
        if self.mode != "downweight":
            return torch.ones_like(snr1)
        return snr_positive_weight(torch.minimum(snr1, snr2), self.full_weight_db, self.zero_weight_db)

    # -- instance discrimination (shared by all modes) ---------------------
    def _one_way(self, query, keys, queue, weights):
        logits = query @ keys.t()
        if queue is not None and queue.numel() > 0:
            logits = torch.cat([logits, query @ queue.t()], dim=1)
        logits = logits / self.temperature

        labels = torch.arange(query.shape[0], device=query.device)
        per_sample = F.cross_entropy(logits, labels, reduction="none")
        loss = (per_sample * weights).sum() / weights.sum().clamp_min(1e-6)
        accuracy = (logits.argmax(dim=1) == labels).float().mean()
        return loss, accuracy

    # -- NRCL noise-aware term --------------------------------------------
    def _noise_aware(self, query, k_pos, k_noisy):
        """One positive (the moderate view) against one negative (the destroyed view).

        -log[ e^{s+/t} / (e^{s+/t} + e^{s-/t}) ] == softplus((s- - s+)/t).
        """
        s_pos = (query * k_pos).sum(dim=1) / self.temperature
        s_neg = (query * k_noisy).sum(dim=1) / self.temperature
        return F.softplus(s_neg - s_pos).mean()

    # -- supervised contrastive -------------------------------------------
    def _supervised(self, query, keys, key_labels, query_labels):
        """SupCon with negatives restricted to other classes.

        For each anchor the denominator holds one positive plus every different-class key,
        so a same-class sample is never pushed away.
        """
        sim = (query @ keys.t()) / self.temperature
        pos_mask = query_labels[:, None] == key_labels[None, :]

        neg_sim = sim.masked_fill(pos_mask, float("-inf"))
        neg_lse = torch.logsumexp(neg_sim, dim=1)                       # (B,)

        # log softmax over {this positive} U {all negatives}
        log_prob = sim - torch.logaddexp(sim, neg_lse[:, None])
        n_pos = pos_mask.sum(dim=1).clamp_min(1)
        loss = -(log_prob * pos_mask).sum(dim=1) / n_pos

        accuracy = pos_mask.gather(1, sim.argmax(dim=1, keepdim=True)).float().mean()
        return loss.mean(), accuracy

    # -- entry point -------------------------------------------------------
    def forward(self, q1, k2, q2, k1, weights, queue=None,
                k_noisy=None, labels=None, queue_labels=None):
        """Symmetric loss; ``q*`` from the online branch, ``k*`` from the EMA teacher."""
        if self.mode == "supervised":
            keys = k2 if queue is None else torch.cat([k2, queue], dim=0)
            key_labels = labels if queue_labels is None else torch.cat([labels, queue_labels], dim=0)
            loss_a, acc_a = self._supervised(q1, keys, key_labels, labels)

            keys = k1 if queue is None else torch.cat([k1, queue], dim=0)
            loss_b, acc_b = self._supervised(q2, keys, key_labels, labels)
            return 0.5 * (loss_a + loss_b), 0.5 * (acc_a + acc_b)

        loss_a, acc_a = self._one_way(q1, k2, queue, weights)
        loss_b, acc_b = self._one_way(q2, k1, queue, weights)
        loss = 0.5 * (loss_a + loss_b)
        accuracy = 0.5 * (acc_a + acc_b)

        if self.mode == "nrcl":
            if k_noisy is None:
                raise ValueError("mode='nrcl' requires the destructive third view k_noisy.")
            noise_term = 0.5 * (self._noise_aware(q1, k2, k_noisy) + self._noise_aware(q2, k1, k_noisy))
            loss = loss + self.nrcl_weight * noise_term

        return loss, accuracy


class _Dummy(nn.Module):
    """Placeholder for the unused `loss` / `metric` slots of DefaultTrainer."""

    init_kwargs: Dict = {}


# ---------------------------------------------------------------------------
# Logging.
# ---------------------------------------------------------------------------
class ContrastiveTensorboardLogger(torch_em.trainer.logger_base.TorchEmLogger):
    """Scalar-only logger: loss, top-1 match rate, curriculum state."""

    def __init__(self, trainer, save_root, **unused_kwargs):
        super().__init__(trainer, save_root)
        from torch.utils.tensorboard import SummaryWriter

        self.log_dir = f"./logs/{trainer.name}" if save_root is None else os.path.join(save_root, "logs", trainer.name)
        os.makedirs(self.log_dir, exist_ok=True)
        self.tb = SummaryWriter(self.log_dir)

    def log_train(self, step, loss, lr, x=None, y=None, prediction=None, log_gradients=False):
        self.tb.add_scalar("train/loss", loss, step)
        self.tb.add_scalar("train/lr", lr, step)

    def log_validation(self, step, metric, loss, x=None, y=None, prediction=None):
        self.tb.add_scalar("validation/loss", loss, step)
        self.tb.add_scalar("validation/metric", metric, step)

    def log_scalar(self, step, tag, value):
        self.tb.add_scalar(tag, value, step)


# ---------------------------------------------------------------------------
# Trainer.
# ---------------------------------------------------------------------------
class MoCoContrastiveTrainer(torch_em.trainer.DefaultTrainer):
    """MoCo-style contrastive pretraining; loaders yield ``(view1, view2, snr1, snr2, label)``.

    Args:
        queue_size: Buffered negative keys, so the negative count is not capped by the batch
            size. 0 disables the queue.
        warmup_momentum: Ramp the momentum in from ``1 - 1/(iter+1)``, so the teacher tracks
            the fast-changing student early on.
    """

    def __init__(
        self,
        model: ContrastiveEncoder,
        contrastive_loss: NoiseAwareInfoNCE,
        momentum: float = 0.99,
        queue_size: int = 4096,
        warmup_momentum: bool = True,
        **kwargs,
    ):
        kwargs.pop("loss", None)
        kwargs.pop("metric", None)
        super().__init__(model=model, loss=_Dummy(), metric=_Dummy(), **kwargs)

        self.contrastive_loss = contrastive_loss
        self.momentum = momentum
        self.queue_size = queue_size
        self.warmup_momentum = warmup_momentum
        self._kwargs = kwargs

        with torch.no_grad():
            self.teacher = deepcopy(self.model)
            self.teacher.predictor = None  # online-only head; never used for the keys
            for param in self.teacher.parameters():
                param.requires_grad = False

        # Match by name, since the teacher has no predictor.
        teacher_params = dict(self.teacher.named_parameters())
        teacher_buffers = dict(self.teacher.named_buffers())
        self._ema_params = [(p, teacher_params[n]) for n, p in self.model.named_parameters() if n in teacher_params]
        self._ema_buffers = [(b, teacher_buffers[n]) for n, b in self.model.named_buffers() if n in teacher_buffers]

        self.register_queue(model.init_kwargs["projection_dim"])

    # -- queue ------------------------------------------------------------
    def register_queue(self, dim: int) -> None:
        self._queue = torch.zeros(self.queue_size, dim) if self.queue_size > 0 else None
        # Labels ride along with the keys so supervised mode can exclude same-class negatives.
        self._queue_labels = torch.full((self.queue_size,), -1, dtype=torch.long) \
            if self.queue_size > 0 else None
        self._queue_ptr = 0
        self._queue_filled = 0

    @property
    def queue(self) -> Optional[torch.Tensor]:
        if self._queue is None or self._queue_filled == 0:
            return None
        return self._queue[:self._queue_filled]

    @property
    def queue_labels(self) -> Optional[torch.Tensor]:
        if self._queue_labels is None or self._queue_filled == 0:
            return None
        return self._queue_labels[:self._queue_filled]

    @torch.no_grad()
    def _enqueue(self, keys: torch.Tensor, labels: Optional[torch.Tensor] = None) -> None:
        if self._queue is None:
            return
        keys = keys.detach().float()
        n = keys.shape[0]
        if labels is None:
            labels = torch.full((n,), -1, dtype=torch.long, device=self._queue_labels.device)
        labels = labels.detach().to(self._queue_labels.device)

        if n >= self.queue_size:
            self._queue.copy_(keys[-self.queue_size:])
            self._queue_labels.copy_(labels[-self.queue_size:])
            self._queue_ptr, self._queue_filled = 0, self.queue_size
            return
        end = self._queue_ptr + n
        if end <= self.queue_size:
            self._queue[self._queue_ptr:end] = keys
            self._queue_labels[self._queue_ptr:end] = labels
        else:
            split = self.queue_size - self._queue_ptr
            self._queue[self._queue_ptr:] = keys[:split]
            self._queue_labels[self._queue_ptr:] = labels[:split]
            self._queue[:end - self.queue_size] = keys[split:]
            self._queue_labels[:end - self.queue_size] = labels[split:]
        self._queue_ptr = end % self.queue_size
        self._queue_filled = min(self.queue_size, self._queue_filled + n)

    # -- teacher ----------------------------------------------------------
    @torch.no_grad()
    def _momentum_update(self) -> None:
        m = min(1 - 1 / (self._iteration + 1), self.momentum) if self.warmup_momentum else self.momentum
        for param, param_teacher in self._ema_params:
            param_teacher.data.mul_(m).add_(param.data, alpha=1.0 - m)
        for buf, buf_teacher in self._ema_buffers:
            buf_teacher.data.copy_(buf.data)

    def _initialize(self, iterations, load_from_checkpoint, epochs=None):
        best_metric = super()._initialize(iterations, load_from_checkpoint, epochs)
        self.teacher.to(self.device)
        if self._queue is not None:
            self._queue = self._queue.to(self.device)
            self._queue_labels = self._queue_labels.to(self.device)
        return best_metric

    def save_checkpoint(self, name, current_metric, best_metric, **extra_save_dict):
        """@private"""
        extra_save_dict.setdefault("teacher_state", self.teacher.state_dict())
        extra_save_dict.setdefault("encoder_kwargs", self.model.init_kwargs)
        super().save_checkpoint(name, current_metric, best_metric, **extra_save_dict)

    def load_checkpoint(self, checkpoint="best"):
        """@private"""
        save_dict = super().load_checkpoint(checkpoint)
        if save_dict is not None and "teacher_state" in save_dict:
            self.teacher.load_state_dict(save_dict["teacher_state"])
            self.teacher.to(self.device)
        return save_dict

    # -- steps ------------------------------------------------------------
    def _step(self, batch, forward_context):
        # 5-tuple normally; 6-tuple when the loss needs the destructive third view (NRCL).
        if len(batch) == 6:
            view1, view2, view3, snr1, snr2, labels = batch
            view3 = view3.to(self.device, non_blocking=True)
        else:
            view1, view2, snr1, snr2, labels = batch
            view3 = None
        view1 = view1.to(self.device, non_blocking=True)
        view2 = view2.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)
        weights = self.contrastive_loss.pair_weights(
            snr1.to(self.device, non_blocking=True), snr2.to(self.device, non_blocking=True)
        )

        with forward_context():
            q1 = self.model(view1, predict=True)
            q2 = self.model(view2, predict=True)
            with torch.no_grad():
                k1 = self.teacher(view1)
                k2 = self.teacher(view2)
                k_noisy = self.teacher(view3) if view3 is not None else None
            loss, accuracy = self.contrastive_loss(
                q1, k2, q2, k1, weights, self.queue,
                k_noisy=k_noisy, labels=labels, queue_labels=self.queue_labels,
            )

        keys = torch.cat([k1, k2], dim=0)
        key_labels = torch.cat([labels, labels], dim=0)
        return loss, accuracy, weights, keys, key_labels

    def _train_epoch_impl(self, progress, forward_context, backprop):
        self.model.train()

        n_iter = 0
        t_per_iter = time.time()
        for batch in self.train_loader:
            # Advance the corruption curriculum in the (shared-memory) dataset.
            dataset = self.train_loader.dataset
            if hasattr(dataset, "set_iteration"):
                dataset.set_iteration(self._iteration)

            self.optimizer.zero_grad()
            loss, accuracy, weights, keys, key_labels = self._step(batch, forward_context)
            backprop(loss)

            self._enqueue(keys, key_labels)
            with torch.no_grad():
                self._momentum_update()

            if self.logger is not None:
                lr = [pm["lr"] for pm in self.optimizer.param_groups][0]
                self.logger.log_train(self._iteration, loss.item(), lr)
                if self._iteration % self.log_image_interval == 0:
                    self.logger.log_scalar(self._iteration, "train/top1_match", accuracy.item())
                    self.logger.log_scalar(self._iteration, "train/mean_pair_weight", weights.mean().item())
                    if hasattr(dataset, "severity"):
                        self.logger.log_scalar(self._iteration, "train/noise_severity", dataset.severity)

            self._iteration += 1
            n_iter += 1
            if self._iteration >= self.max_iteration:
                break
            progress.update(1)

        t_per_iter = (time.time() - t_per_iter) / max(n_iter, 1)
        return t_per_iter

    def _validate_impl(self, forward_context):
        # The teacher must switch with the student: leaving it in train mode makes the keys use
        # batch statistics while the queries use running ones, which destroys the match.
        self.model.eval()
        self.teacher.eval()

        loss_val, acc_val, n = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in self.val_loader:
                loss, accuracy, _, _, _ = self._step(batch, forward_context)
                loss_val += loss.item()
                acc_val += accuracy.item()
                n += 1

        self.teacher.train()

        loss_val /= max(n, 1)
        acc_val /= max(n, 1)
        # Lower is better, to match ReduceLROnPlateau(mode="min") and the torch_em convention.
        metric = 1.0 - acc_val
        if self.logger is not None:
            self.logger.log_validation(self._iteration, metric, loss_val)
        return metric


# ---------------------------------------------------------------------------
# Transferring the pretrained backbone into the classifier.
# ---------------------------------------------------------------------------
def extract_backbone_state_dict(checkpoint_path: str, prefer_teacher: bool = False) -> Dict[str, torch.Tensor]:
    """Pull the plain ResNet3d weights out of a contrastive checkpoint.

    Args:
        checkpoint_path: A ``best.pt`` / ``latest.pt``, or the directory holding it.
        prefer_teacher: Use the EMA teacher weights; smoother, often the better init.
    """
    if os.path.isdir(checkpoint_path):
        checkpoint_path = os.path.join(checkpoint_path, "best.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    key = "teacher_state" if (prefer_teacher and "teacher_state" in checkpoint) else "model_state"
    state = checkpoint[key]

    prefix = "backbone."
    backbone_state = {
        k[len(prefix):]: v for k, v in state.items()
        if k.startswith(prefix) and not k[len(prefix):].startswith("fc.")
    }
    if not backbone_state:
        raise ValueError(f"No 'backbone.*' weights found in {checkpoint_path} (key '{key}').")
    return backbone_state


def load_pretrained_backbone(model: nn.Module, checkpoint_path: str, prefer_teacher: bool = False) -> nn.Module:
    """Init a classifier backbone from a contrastive checkpoint; ``fc`` stays random."""
    backbone_state = extract_backbone_state_dict(checkpoint_path, prefer_teacher=prefer_teacher)
    missing, unexpected = model.load_state_dict(backbone_state, strict=False)
    missing = [k for k in missing if not k.startswith("fc.")]
    print(
        f"Loaded contrastive backbone from {checkpoint_path} "
        f"({'teacher' if prefer_teacher else 'online'} weights): "
        f"{len(backbone_state)} tensors, {len(missing)} missing, {len(unexpected)} unexpected."
    )
    if unexpected:
        print(f"  unexpected keys (first 5): {unexpected[:5]}")
    if missing:
        print(f"  missing keys (first 5): {missing[:5]}")
    return model


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------
def get_contrastive_loader(
    paths: List[str],
    target_root: str,
    max_extent: int,
    batch_size: int,
    in_channels: int = 1,
    normalization: Optional[Callable] = None,
    view_augmentation: Optional[NoiseAwareViewAugment] = None,
    paired_dataset_roots: Optional[Sequence[str]] = None,
    cross_source_prob: float = 0.5,
    n_samples: Optional[int] = None,
    curriculum_iterations: int = 0,
    num_workers: int = 8,
    shuffle: bool = True,
    noisy_view_snr_db: Optional[float] = None,
) -> torch.utils.data.DataLoader:
    ds = ContrastivePairDataset(
        paths=paths,
        target_root=target_root,
        max_extent=max_extent,
        in_channels=in_channels,
        normalization=normalization,
        view_augmentation=view_augmentation,
        paired_dataset_roots=paired_dataset_roots,
        cross_source_prob=cross_source_prob,
        n_samples=n_samples,
        curriculum_iterations=curriculum_iterations,
        noisy_view_snr_db=noisy_view_snr_db,
    )
    # drop_last: the InfoNCE denominator degenerates for a size-1 trailing batch.
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        persistent_workers=False, drop_last=True,
    )
    loader.shuffle = shuffle
    return loader


def contrastive_pretraining(
    name: str,
    train_paths: List[str],
    val_paths: List[str],
    target_root: str,
    max_extent: int,
    in_channels: int = 1,
    batch_size: int = 64,
    lr: float = 1e-3,
    n_iterations: int = int(2e4),
    projection_dim: int = 128,
    hidden_dim: int = 1024,
    use_predictor: bool = True,
    temperature: float = 0.2,
    momentum: float = 0.99,
    queue_size: int = 4096,
    full_weight_db: float = 3.0,
    zero_weight_db: float = -6.0,
    negatives: str = "downweight",
    nrcl_weight: float = 1.0,
    noisy_view_snr_db: float = -12.0,
    curriculum_fraction: float = 0.5,
    normalization: Optional[Callable] = None,
    view_augmentation: Optional[NoiseAwareViewAugment] = None,
    paired_dataset_roots: Optional[Sequence[str]] = None,
    cross_source_prob: float = 0.5,
    n_samples_train: Optional[int] = None,
    n_samples_val: Optional[int] = None,
    num_workers: int = 8,
    save_root: Optional[str] = None,
    check: bool = False,
) -> str:
    """Run stage 1: contrastive pretraining of the classifier's encoder.

    Args:
        target_root: Root of the pick JSONs, used for the crop coordinates.
        max_extent: Crop size without halo, as in supervised training.
        queue_size: Buffered negatives; 0 uses in-batch negatives only.
        full_weight_db / zero_weight_db: SNR band over which a positive pair is down-weighted.
        negatives: What counts as a negative -- "downweight" (ours), "nrcl" (destroyed view of
            the same crop becomes a negative), or "supervised" (same-class keys are positives).
        noisy_view_snr_db: SNR of the destructive third view used by "nrcl".
        curriculum_fraction: Fraction of training over which corruption severity ramps from
            mild to full. 0 disables the curriculum.
        paired_dataset_roots: Dataset dirs with the same tomograms at a different SNR, enabling
            real "same particle, different SNR" positives.
        check: Only inspect a few batches and return.

    Returns:
        The checkpoint folder of the pretrained encoder.
    """
    curriculum_iterations = int(curriculum_fraction * n_iterations)
    loss_fn = NoiseAwareInfoNCE(temperature, full_weight_db, zero_weight_db,
                                mode=negatives, nrcl_weight=nrcl_weight)
    extra_view = noisy_view_snr_db if loss_fn.needs_noisy_view else None
    print(f"Negatives: {negatives}"
          + (f" (destructive view at {noisy_view_snr_db} dB, weight {nrcl_weight})" if extra_view else ""))

    train_loader = get_contrastive_loader(
        train_paths, target_root, max_extent, batch_size, in_channels, normalization,
        view_augmentation, paired_dataset_roots, cross_source_prob,
        n_samples_train, curriculum_iterations, num_workers, noisy_view_snr_db=extra_view,
    )
    val_loader = get_contrastive_loader(
        val_paths, target_root, max_extent, batch_size, in_channels, normalization,
        view_augmentation, paired_dataset_roots, cross_source_prob,
        n_samples_val, 0, num_workers, shuffle=False, noisy_view_snr_db=extra_view,
    )

    if check:
        for i, batch in enumerate(train_loader):
            v1, v2, s1, s2, y = (batch[0], batch[1], batch[-3], batch[-2], batch[-1])
            print(f"batch {i}: {len(batch)} tensors, views {tuple(v1.shape)} / {tuple(v2.shape)}, "
                  f"snr1 {s1.min():.1f}..{s1.max():.1f} dB, labels {y.tolist()[:8]}")
            if i >= 2:
                break
        return ""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ContrastiveEncoder(
        in_channels=in_channels, projection_dim=projection_dim,
        hidden_dim=hidden_dim, use_predictor=use_predictor,
    ).to(device)
    print(f"Contrastive encoder on device: {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    trainer = MoCoContrastiveTrainer(
        name=name,
        model=model,
        contrastive_loss=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        device=device,
        momentum=momentum,
        queue_size=queue_size,
        mixed_precision=True,
        log_image_interval=100,
        compile_model=False,
        logger=ContrastiveTensorboardLogger,
        save_root=save_root,
    )
    trainer.fit(n_iterations)

    print(f"Pretrained encoder written to {trainer.checkpoint_folder}")
    return trainer.checkpoint_folder
