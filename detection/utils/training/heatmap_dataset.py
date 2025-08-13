import os
import warnings
import numpy as np
from typing import List, Union, Tuple, Optional, Any, Callable

import torch

from torch_em.util import ensure_spatial_array, ensure_tensor_with_channels, ensure_patch_shape
from ..image import load_data

from detection.data_processing.create_heatmap import get_label

class HeatmapDataset(torch.utils.data.Dataset):
    max_sampling_attempts = 500
    """The maximal number of sampling attempts, for loading a sample via `__getitem__`.
    This is used when `sampler` rejects a sample, to avoid an infinite loop if no valid sample can be found.
    """

    @staticmethod
    def compute_len(shape, patch_shape):
        if patch_shape is None:
            return 1
        else:
            n_samples = int(np.prod([float(sh / csh) for sh, csh in zip(shape, patch_shape)]))
            return n_samples

    def __init__(
        self,
        raw_path: Union[List[Any], str, os.PathLike],
        raw_key: Optional[str],
        label_path: Union[List[Any], str, os.PathLike],
        patch_shape: Tuple[int, ...],
        raw_transform: Optional[Callable] = None,
        label_transform: Optional[Callable] = None,
        label_transform2: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        dtype: torch.dtype = torch.float32,
        label_dtype: torch.dtype = torch.float32,
        n_samples: Optional[int] = None,
        sampler: Optional[Callable] = None,
        ndim: Optional[int] = None,
        with_channels: bool = False,
        with_label_channels: bool = False,
        with_padding: bool = True,
        z_ext: Optional[int] = None,
        eps: Optional[float] = 10**-8,
        sigma: Optional[float] = None,
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
    ):
        self.raw_path = raw_path
        self.raw_key = raw_key
        #print(f"raw_path {raw_path}, raw_key {raw_key}")
        self.raw = load_data(raw_path, raw_key)
        #print(f"self.raw {self.raw}")

        self.label_path = label_path
        #print(f"self.label_path {self.label_path}")

        self._with_channels = with_channels
        self._with_label_channels = with_label_channels

        shape_raw = self.raw.shape[1:] if self._with_channels else self.raw.shape
        #print(f"shape_raw {shape_raw}")

        self.shape = shape_raw

        self._ndim = len(shape_raw) if ndim is None else ndim
        assert self._ndim in (2, 3, 4), f"Invalid data dimensions: {self._ndim}. Only 2d, 3d or 4d data is supported"

        if patch_shape is not None:
            assert len(patch_shape) in (self._ndim, self._ndim + 1), f"{patch_shape}, {self._ndim}"
            #for raw_path being a list of paths, not just one path
            '''expected_ndim = len(shape_raw) - 1  # Extract ndim from shape_raw
            assert len(patch_shape) in (expected_ndim, expected_ndim + 1), \
                f"Invalid patch_shape {patch_shape}, expected dimensions: {expected_ndim} or {expected_ndim + 1}"'''


        self.patch_shape = patch_shape

        self.raw_transform = raw_transform
        self.label_transform = label_transform
        self.label_transform2 = label_transform2
        self.transform = transform
        self.sampler = sampler
        self.with_padding = with_padding

        self.dtype = dtype
        self.label_dtype = label_dtype

        self._len = self.compute_len(self.shape, self.patch_shape) if n_samples is None else n_samples

        self.z_ext = z_ext

        self.sample_shape = patch_shape
        self.trafo_halo = None

        # Julias arguments!
        self.eps = eps
        self.sigma = sigma
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound


    def __len__(self):
        return self._len

    @property
    def ndim(self):
        return self._ndim

    def _sample_bounding_box(self):
        if self.sample_shape is None:
            if self.z_ext is None:
                bb_start = [0] * len(self.shape)
                patch_shape_for_bb = self.shape
            else:
                z_diff = self.shape[0] - self.z_ext
                bb_start = [np.random.randint(0, z_diff) if z_diff > 0 else 0] + [0] * len(self.shape[1:])
                patch_shape_for_bb = (self.z_ext, *self.shape[1:])

        else:
            bb_start = [
                np.random.randint(0, sh - psh) if sh - psh > 0 else 0 for sh, psh in zip(self.shape, self.sample_shape)
            ]
            patch_shape_for_bb = self.sample_shape

        return tuple(slice(start, start + psh) for start, psh in zip(bb_start, patch_shape_for_bb))

    def _get_desired_raw_and_labels(self):
        bb = self._sample_bounding_box()
        bb_raw = (slice(None),) + bb if self._with_channels else bb
        bb_labels = (slice(None),) + bb if self._with_label_channels else bb
        raw = self.raw[bb_raw]
        labels = get_label(
            self.label_path, self.shape, eps=self.eps, sigma=self.sigma,
            lower_bound=self.lower_bound, upper_bound=self.upper_bound, bb=bb_labels
        )
        return raw, labels

    def _get_sample(self, index):
        if self.raw is None or self.label_path is None:
            raise RuntimeError("SegmentationDataset has not been properly deserialized.")

        raw, labels = self._get_desired_raw_and_labels()

        if self.sampler is not None:
            sample_id = 0
            while not self.sampler(raw, labels):
                raw, labels = self._get_desired_raw_and_labels()
                sample_id += 1
                if sample_id > self.max_sampling_attempts:
                    raise RuntimeError(f"Could not sample a valid batch in {self.max_sampling_attempts} attempts")

        # Padding the patch to match the expected input shape.
        if self.patch_shape is not None and self.with_padding:
            raw, labels = ensure_patch_shape(
                raw=raw,
                labels=labels,
                patch_shape=self.patch_shape,
                have_raw_channels=self._with_channels,
                have_label_channels=self._with_label_channels,
            )

        # squeeze the singleton spatial axis if we have a spatial shape that is larger by one than self._ndim
        if self.patch_shape is not None and len(self.patch_shape) == self._ndim + 1:
            raw = raw.squeeze(1 if self._with_channels else 0)
            labels = labels.squeeze(1 if self._with_label_channels else 0)

        return raw, labels

    def crop(self, tensor):
        """@private
        """
        bb = self.inner_bb
        if tensor.ndim > len(bb):
            bb = (tensor.ndim - len(bb)) * (slice(None),) + bb
        return tensor[bb]

    def __getitem__(self, index):
        raw, labels = self._get_sample(index)
        initial_label_dtype = labels.dtype

        if self.raw_transform is not None:
            raw = self.raw_transform(raw)

        if self.label_transform is not None:
            labels = self.label_transform(labels)

        if self.transform is not None:
            raw, labels = self.transform(raw, labels)
            if self.trafo_halo is not None:
                raw = self.crop(raw)
                labels = self.crop(labels)

        # support enlarging bounding box here as well (for affinity transform) ?
        if self.label_transform2 is not None:
            labels = ensure_spatial_array(labels, self.ndim, dtype=initial_label_dtype)
            labels = self.label_transform2(labels)

        raw = ensure_tensor_with_channels(raw, ndim=self._ndim, dtype=self.dtype)
        labels = ensure_tensor_with_channels(labels, ndim=self._ndim, dtype=self.label_dtype)
        return raw, labels