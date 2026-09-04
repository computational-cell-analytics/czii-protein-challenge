"""Positive pairs of subtomogram views for contrastive pretraining.

Two ways of building a pair, which can be mixed:

1. Simulated corruption -- one crop, two independent draws from :class:`NoiseAwareViewAugment`.
2. Real cross-SNR pairing -- the same particle from two datasets simulated from the same
   ground truth at different noise settings (e.g. ``ExperimentRuns_faket_snr_0_12_0_2`` and
   ``ExperimentRuns_basic_snr_0_1_0_15``, which share tomogram names and pick coordinates).
   The ``x_a = C_a(s), x_b = C_b(s)`` experiment with real physics.

Items are ``(view1, view2, snr1, snr2, label)``; the label is unused by the contrastive loss.
"""

import os
import random
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from classification.config import MAX_EXTENT_HALO
from classification.training import get_coords_and_targets

from .noise_augmentation import NoiseAwareViewAugment
from .volume_access import bbox_size_for, load_crop, volume_shape


class ContrastivePairDataset(torch.utils.data.Dataset):
    """Yields two noise-augmented views of the same subtomogram.

    Args:
        target_root: Root of the pick JSONs; used only for the coordinates and labels.
        normalization: Applied to the raw crop before the corruption operators.
        paired_dataset_roots: Dataset dirs holding the same tomogram names at a different noise
            realization; ``cross_source_prob`` of the pairs are then drawn from two of them.
        n_samples: Optional fixed epoch length (samples are drawn cyclically).
        curriculum_iterations: Iterations over which corruption severity ramps 0 -> 1.
            0 means always full severity.
        noisy_view_snr_db: If set, also emit a third view at this fixed (destructive) SNR.
            Items become ``(v1, v2, v3, snr1, snr2, label)``. Used by the NRCL variant.
    """

    def __init__(
        self,
        paths: List[str],
        target_root: str,
        max_extent: int = 39,
        in_channels: int = 1,
        normalization: Optional[Callable] = None,
        view_augmentation: Optional[NoiseAwareViewAugment] = None,
        paired_dataset_roots: Optional[Sequence[str]] = None,
        cross_source_prob: float = 0.5,
        n_samples: Optional[int] = None,
        curriculum_iterations: int = 0,
        halo: int = MAX_EXTENT_HALO,
        noisy_view_snr_db: Optional[float] = None,
    ):
        self.max_extent = max_extent
        self.in_channels = in_channels
        self.normalization = normalization
        self.view_augmentation = view_augmentation or NoiseAwareViewAugment()
        self.cross_source_prob = cross_source_prob
        self.halo = halo
        self._n_samples = n_samples
        self.curriculum_iterations = curriculum_iterations
        self.noisy_view_snr_db = noisy_view_snr_db

        # Shared memory, so the trainer's curriculum updates reach the workers.
        self._severity = torch.zeros(1, dtype=torch.float32).share_memory_()
        if curriculum_iterations <= 0:
            self._severity.fill_(1.0)

        coords_list, targets_list = get_coords_and_targets(paths, target_root=target_root)

        half = bbox_size_for(max_extent, halo) // 2
        self.sources: List[List[str]] = []
        self.coords: List[Tuple[int, int, int]] = []
        self.targets: List[str] = []

        n_cross = 0
        for path, coords, targets in zip(paths, coords_list, targets_list):
            shape = volume_shape(path)
            siblings = self._find_siblings(path, paired_dataset_roots)
            n_cross += len(siblings) - 1
            for c, t in zip(coords, targets):
                z, y, x = c
                if (z - half >= 0 and z + half < shape[0] and
                        y - half >= 0 and y + half < shape[1] and
                        x - half >= 0 and x + half < shape[2]):
                    self.sources.append(siblings)
                    self.coords.append((z, y, x))
                    self.targets.append(t)

        if len(self.coords) == 0:
            raise ValueError("No in-bounds crops found for contrastive pretraining.")

        self.classes = sorted(set(self.targets))
        self.label_to_index = {label: idx for idx, label in enumerate(self.classes)}

        print(
            f"Contrastive dataset: {len(self.coords)} crops from {len(paths)} tomograms"
            + (f", {n_cross} extra cross-SNR sources" if n_cross else "")
        )

    @staticmethod
    def _find_siblings(path: str, paired_dataset_roots: Optional[Sequence[str]]) -> List[str]:
        """Directories holding the same tomogram at a different noise realization."""
        sources = [path]
        if not paired_dataset_roots:
            return sources
        name = os.path.basename(os.path.normpath(path))
        for root in paired_dataset_roots:
            candidate = os.path.join(root, name)
            if os.path.isdir(candidate) and os.path.normpath(candidate) != os.path.normpath(path):
                sources.append(candidate)
        return sources

    # -- curriculum -------------------------------------------------------
    def set_iteration(self, iteration: int) -> None:
        """Advance the corruption curriculum; called by the trainer every iteration."""
        if self.curriculum_iterations > 0:
            self._severity.fill_(min(1.0, iteration / float(self.curriculum_iterations)))

    @property
    def severity(self) -> float:
        return float(self._severity.item())

    # -- data -------------------------------------------------------------
    def __len__(self):
        return self._n_samples if self._n_samples is not None else len(self.coords)

    def _load(self, source: str, coord) -> torch.Tensor:
        x = load_crop(source, coord, self.max_extent, self.in_channels, self.halo)
        if self.normalization is not None:
            x = self.normalization(x)
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        return x.float()

    def _pick_sources(self, sources: List[str]) -> Tuple[str, str]:
        if len(sources) > 1 and random.random() < self.cross_source_prob:
            a, b = random.sample(sources, 2)
            return a, b
        s = random.choice(sources)
        return s, s

    def __getitem__(self, index):
        real_index = index % len(self.coords)
        coord = self.coords[real_index]
        sources = self.sources[real_index]
        label = self.label_to_index[self.targets[real_index]]

        src1, src2 = self._pick_sources(sources)
        try:
            x1 = self._load(src1, coord)
            x2 = x1 if src1 == src2 else self._load(src2, coord)
        except (IndexError, ValueError, KeyError):
            return self[random.randint(0, len(self.coords) - 1)]

        severity = self.severity
        view1, snr1 = self.view_augmentation(x1.clone(), severity)
        view2, snr2 = self.view_augmentation(x2.clone(), severity)
        label_t = torch.tensor(label, dtype=torch.long)
        snr1_t = torch.tensor(snr1, dtype=torch.float32)
        snr2_t = torch.tensor(snr2, dtype=torch.float32)

        if self.noisy_view_snr_db is None:
            return view1, view2, snr1_t, snr2_t, label_t

        # NRCL: a third view at an information-destroying SNR, used as an explicit negative.
        view3, _ = self.view_augmentation(x1.clone(), 1.0, force_snr_db=self.noisy_view_snr_db)
        return view1, view2, view3, snr1_t, snr2_t, label_t


class SingleViewDataset(torch.utils.data.Dataset):
    """Labeled crops with an optional fixed transform; used by the noise-robustness eval."""

    def __init__(
        self,
        paths: List[str],
        target_root: str,
        max_extent: int = 39,
        in_channels: int = 1,
        normalization: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        label_to_index: Optional[dict] = None,
        halo: int = MAX_EXTENT_HALO,
    ):
        self.max_extent = max_extent
        self.in_channels = in_channels
        self.normalization = normalization
        self.transform = transform
        self.halo = halo

        coords_list, targets_list = get_coords_and_targets(paths, target_root=target_root)
        half = bbox_size_for(max_extent, halo) // 2

        self.paths, self.coords, self.targets = [], [], []
        for path, coords, targets in zip(paths, coords_list, targets_list):
            shape = volume_shape(path)
            for c, t in zip(coords, targets):
                z, y, x = c
                if (z - half >= 0 and z + half < shape[0] and
                        y - half >= 0 and y + half < shape[1] and
                        x - half >= 0 and x + half < shape[2]):
                    self.paths.append(path)
                    self.coords.append((z, y, x))
                    self.targets.append(t)

        self.classes = sorted(set(self.targets))
        self.label_to_index = label_to_index or {label: idx for idx, label in enumerate(self.classes)}

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, index):
        try:
            x = load_crop(self.paths[index], self.coords[index], self.max_extent, self.in_channels, self.halo)
        except (IndexError, ValueError):
            return self[random.randint(0, len(self.coords) - 1)]
        if self.normalization is not None:
            x = self.normalization(x)
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        x = x.float()
        if self.transform is not None:
            x = self.transform(x)
        label = self.label_to_index.get(self.targets[index], -1)
        return x, torch.tensor(label, dtype=torch.long)
