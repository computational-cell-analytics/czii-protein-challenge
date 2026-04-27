import os
import numpy as np
from typing import List, Union, Tuple, Optional, Any, Callable

import torch

from torch_em.util import ensure_spatial_array, ensure_tensor_with_channels, ensure_patch_shape
from torch_em.transform.raw import standardize
from ..image import load_data


class DetectionDataset(torch.utils.data.Dataset):
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
        with_channels: bool = True,
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
        self.raw = load_data(raw_path, raw_key)

        self.label_path = label_path
        self._with_channels = with_channels
        self._with_label_channels = with_label_channels

        shape_raw = self.raw.shape[1:] if self._with_channels else self.raw.shape
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
        self.raw_transform = raw_transform if raw_transform is not None else standardize
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

        # Extra args
        self.eps = eps
        self.sigma = sigma
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.halo = 10

    def __len__(self):
        return self._len

    @property
    def ndim(self):
        return self._ndim

    def _sample_bounding_box(self):
        """
        Sample a random bounding box, ensuring that a halo can be added
        without exceeding the image boundaries.
        """
        if self.sample_shape is None:
            # Patch shape is full image (or z_ext for z-axis)
            if self.z_ext is None:
                bb_start = [0] * len(self.shape)
                patch_shape_for_bb = self.shape
            else:
                z_diff = self.shape[0] - self.z_ext - self.halo * 2
                bb_start = [np.random.randint(0, z_diff) if z_diff > 0 else 0] + [0] * len(self.shape[1:])
                patch_shape_for_bb = (self.z_ext, *self.shape[1:])
        else:
            bb_start = []
            for sh, psh in zip(self.shape, self.sample_shape):
                max_start = max(0, sh - psh - 2 * self.halo)  # reserve space for halo on both sides
                start = np.random.randint(0, max_start + 1) if max_start > 0 else 0
                bb_start.append(start)
            patch_shape_for_bb = self.sample_shape

        return tuple(slice(start, start + psh) for start, psh in zip(bb_start, patch_shape_for_bb))

    def _get_desired_raw_and_labels(self):
        bb = self._sample_bounding_box()
        # Extend the BB with halo
        if self.halo > 0:
            bb_halo = []
            for i, s in enumerate(bb):
                start = max(0, s.start - self.halo)
                stop = min(self.shape[i], s.stop + self.halo)
                bb_halo.append(slice(start, stop))
            bb_for_loading = tuple(bb_halo)
        else:
            bb_for_loading = bb

        # Add channel slices if needed
        bb_raw = (slice(None),) + bb_for_loading if self._with_channels else bb_for_loading
        bb_labels = (slice(None),) + bb_for_loading if self._with_label_channels else bb_for_loading

        raw = self.raw[bb_raw]

        #get labels
        labels = self.label_transform(
            self.label_path,
            self.shape,
            bb_labels,
            bb_for_loading,
        )

        #Crop out the halo so all outputs match original bounding box
        if self.halo > 0:
            # Compute slices for cropping the halo
            slices_crop = tuple(
                slice(self.halo, dim - self.halo) if dim > 2 * self.halo else slice(None)
                for dim in labels.shape[-3:]
            )

            # Crop raw
            if raw.ndim == 4:  # (C, Z, Y, X)
                raw = raw[:, slices_crop[0], slices_crop[1], slices_crop[2]]
            else:  # (Z, Y, X)
                raw = raw[slices_crop[0], slices_crop[1], slices_crop[2]]

            # Crop labels
            if labels.ndim == 4:  # (C, Z, Y, X)
                labels = labels[:, slices_crop[0], slices_crop[1], slices_crop[2]]

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
                raw=raw, labels=labels,
                patch_shape=self.patch_shape,
                have_raw_channels=self._with_channels,
                have_label_channels=(labels.ndim == self._ndim + 1),
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
        if self.transform is not None:
            raw, labels = self.transform(raw, labels)
            if self.trafo_halo is not None:
                raw = self.crop(raw)
                labels = self.crop(labels)

        if self.label_transform2 is not None:
            labels = ensure_spatial_array(labels, self.ndim, dtype=initial_label_dtype)
            labels = self.label_transform2(labels)

        raw = ensure_tensor_with_channels(raw, ndim=self._ndim, dtype=self.dtype)
        labels = ensure_tensor_with_channels(labels, ndim=self._ndim, dtype=self.label_dtype)

        '''visualize = True  # set to True to open Napari
        if visualize:
            try:
                import napari

                # Convert torch tensors to numpy for visualization
                raw_np = raw.detach().cpu().numpy().squeeze()
                labels_np = labels.detach().cpu().numpy()

                # Extract the heatmap (channel 0) and flow (channels 1-3 or 1-4)
                heatmap_np = labels_np[0]
                flow_np = labels_np[1:4]  # only first 3 flow components

                viewer = napari.Viewer()
                viewer.add_image(raw_np, name='raw', contrast_limits=[raw_np.min(), raw_np.max()])
                viewer.add_image(heatmap_np, name='heatmap', colormap='magenta', opacity=0.6)

                # Add flow visualization (as vectors)
                # Flow shape: (3, z, y, x)
                flow_trans = np.moveaxis(flow_np, 0, -1)  # (z, y, x, 3)

                # Replace NaN or inf values
                flow_trans = np.nan_to_num(flow_trans, nan=0.0, posinf=0.0, neginf=0.0)

                if flow_trans.ndim == 4:
                    viewer.add_vectors(
                        flow_trans[::4, ::4, ::4, :],  # downsample for readability
                        name='stereographic_flow',
                        edge_width=0.4,
                        length=4,
                        opacity=0.7, scale = (4, 4, 4)
                    )
                #, scale = (1.0012, 1.0012, 1.0012)  # (Z, Y, X) spacing
                print(f"raw_np shape: {raw_np.shape}")
                print(f"heatmap_np shape: {heatmap_np.shape}")
                print(f"flow_np shape: {flow_np.shape}")
                print(f"flow_trans shape: {flow_trans.shape}")


                napari.run()

            except ImportError:
                warnings.warn("Napari is not installed. Visualization skipped.")
            except Exception as e:
                warnings.warn(f"Napari visualization failed: {e}")'''

        return raw, labels
