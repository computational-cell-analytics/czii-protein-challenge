import os
import warnings
import numpy as np
from typing import List, Union, Tuple, Optional, Any, Callable

import torch

from torch_em.util import ensure_spatial_array, ensure_tensor_with_channels, ensure_patch_shape
from ..image import load_data
from detection.data_processing.create_heatmap import get_label, parse_json_files
from external.spotiflow.spotiflow.utils.peaks import points_to_flow3d
import napari

def compute_stereographic_flow(
    coords: np.ndarray,
    patch_shape: Tuple[int, int, int],
    bb: Optional[Tuple[slice, ...]] = None,
    sigma: float = 1.5,
    grid: Union[int, Tuple[int, int, int]] = (1, 1, 1),
) -> np.ndarray:
    """
    Compute stereographic flow for a patch.

    Parameters
    ----------
    coords : np.ndarray
        Nx3 array of absolute (z, y, x) coordinates of points in the full image.
    patch_shape : tuple(int,int,int)
        (z, y, x) shape in pixels of the output patch where flow will be computed.
    bb : tuple(slice,...), optional
        If provided, the bounding-box (in full-image coordinates) used to sample the patch.
        The function will offset coords into patch-local coordinates by subtracting bb.start values.
    sigma : float
        stereographic scale passed to points_to_flow3d
    grid : int or tuple(int,int,int)
        grid spacing for the flow (passed to points_to_flow3d)

    Returns
    -------
    flow : np.ndarray
        Array with shape (4, Z, Y, X) and dtype float32 where the channel order is
        [w', z', y', x'] (same as spotiflow).
    """
    # coords may be empty
    if coords is None:
        coords = np.zeros((0, 3), dtype=np.float32)
    coords = np.asarray(coords, dtype=np.float32)

    # If we have a bounding box, shift coordinates into patch-local coordinates and filter
    if bb is not None:
        # bb is expected to be a tuple of slices (z_slice, y_slice, x_slice)
        # If bb has channel dimension included (slice(None), ...) try to drop it
        # Find first slice with non-None start that looks like spatial slice
        # We'll assume bb contains only spatial slices here
        starts = []
        stops = []
        for s in bb:
            if isinstance(s, slice):
                start = 0 if s.start is None else s.start
                stop = None if s.stop is None else s.stop
                starts.append(start)
                stops.append(stop)
        if len(starts) >= 3:
            # take last 3 slices if extra channel slice was prepended
            starts = starts[-3:]
            stops = stops[-3:]
        elif len(starts) != 3:
            raise ValueError("bb must contain 3 spatial slices (z,y,x) or channel plus 3 slices.")

        starts = np.array(starts, dtype=np.float32)
        stops = np.array(stops, dtype=np.float32)

        # Shift coords into patch-local coordinates
        if coords.shape[0] > 0:
            coords_local = coords - starts[np.newaxis, :]
            # Keep points that fall inside the patch bounds [0, size)
            inside_mask = np.all((coords_local >= 0) & (coords_local < (stops - starts)), axis=1)
            coords_local = coords_local[inside_mask]
        else:
            coords_local = coords.copy()
    else:
        coords_local = coords.copy()

    # points_to_flow3d expects points as float32 array shape (N,3) with coords in (z,y,x)
    # patch_shape is (Z, Y, X)
    if isinstance(grid, int):
        grid_arg = (grid, grid, grid)
    else:
        grid_arg = grid

    # If no points left, points_to_flow3d will return zeros with w=-1 (as in spotiflow)
    flow_np = points_to_flow3d(coords_local.astype(np.float32, copy=False), tuple(patch_shape), sigma=sigma, grid=grid_arg)

    # spotiflow returns (Z, Y, X, 4) and callers transpose to (4, Z, Y, X)
    # Here we transpose and return (4, Z, Y, X) as float32
    flow_np = np.transpose(flow_np, (3, 0, 1, 2)).astype(np.float32, copy=False)
    return flow_np


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
        print(f"self.raw.shape {self.raw.shape}")
        print(f"shape_raw {shape_raw}")
        print(f"ndim{ndim}")
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

        # Extra args
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

        # --- Load and compute heatmap ---
        heatmap = get_label(
            self.label_path, self.shape, eps=self.eps, sigma=self.sigma,
            lower_bound=self.lower_bound, upper_bound=self.upper_bound, bb=bb_labels
        )

        # --- Load JSON coords for flow computation ---
        picks_folder = os.path.join(self.label_path, "Picks")
        json_files = [os.path.join(picks_folder, f) for f in os.listdir(picks_folder) if f.endswith(".json")]
        coords, _ = parse_json_files(json_files)

        # Compute the stereographic flow for the sampled patch.
        # Determine the patch shape (z,y,x) used by the heatmap (heatmap might be channel-first or not)
        # heatmap coming from get_label likely has shape (Z,Y,X) or (1,Z,Y,X). Normalize:
        heatmap_arr = np.asarray(heatmap)
        if heatmap_arr.ndim == 4:
            # assume (C,Z,Y,X) or (1,Z,Y,X)
            # extract spatial shape ignoring channel dimension(s)
            patch_spatial_shape = heatmap_arr.shape[-3:]
            heatmap_spatial = heatmap_arr.squeeze(0)  # becomes (Z,Y,X)
        elif heatmap_arr.ndim == 3:
            patch_spatial_shape = heatmap_arr.shape
            heatmap_spatial = heatmap_arr
        else:
            # unexpected but attempt fallback
            patch_spatial_shape = tuple(self.patch_shape) if self.patch_shape is not None else heatmap_arr.shape[-3:]
            heatmap_spatial = heatmap_arr.reshape(patch_spatial_shape)

        # call compute_stereographic_flow with the bounding box corresponding to spatial slices (not channel slice)
        # The bb returned earlier is spatial slices; in case bb_labels had a leading channel slice, pass the spatial bb.
        spatial_bb = bb if (len(bb) == 3 and all(isinstance(s, slice) for s in bb)) else tuple(bb[-3:])

        flow = compute_stereographic_flow(coords, patch_spatial_shape, bb=spatial_bb, sigma=self.sigma if self.sigma is not None else 1.5, grid=(1, 1, 1))

        # Combine heatmap and flow into a single labels array:
        # heatmap -> channel 0, flow channels 1..4
        # ensure shapes align: heatmap_spatial (Z,Y,X), flow (4,Z,Y,X)
        # Expand heatmap to single channel first
        heatmap_channel = heatmap_spatial[np.newaxis, ...].astype(np.float32, copy=False)
        labels_combined = np.concatenate([heatmap_channel, flow], axis=0)  # shape (1+4, Z, Y, X)

        '''viewer = napari.Viewer(title="Patch visualization")

        # raw may have a channel dimension; squeeze if needed
        raw_disp = np.asarray(raw.squeeze())
        viewer.add_image(raw_disp, name="raw", contrast_limits=(np.percentile(raw_disp, 1), np.percentile(raw_disp, 99)))

        viewer.add_image(heatmap_spatial, name="heatmap", colormap="inferno", blending="additive")

        # Flow visualization: show each flow component (w', z', y', x')
        for i, ch in enumerate(["w'", "z'", "y'", "x'"]):
            viewer.add_image(flow[i], name=f"flow_{ch}", colormap="turbo", blending="additive")

        napari.run()'''

        return raw, labels_combined

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
                have_label_channels=True, #self._with_label_channels was previously and gives error now, see jobID 11321617 TODO
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

        #TODO do I need this?!
        # support enlarging bounding box here as well (for affinity transform) ?
        if self.label_transform2 is not None:
            labels = ensure_spatial_array(labels, self.ndim, dtype=initial_label_dtype)
            labels = self.label_transform2(labels)


        raw = ensure_tensor_with_channels(raw, ndim=self._ndim, dtype=self.dtype)
        #TODO what is right?
        labels = ensure_tensor_with_channels(labels, ndim=self._ndim, dtype=self.label_dtype)
        #labels = torch.as_tensor(labels, dtype=self.label_dtype)

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