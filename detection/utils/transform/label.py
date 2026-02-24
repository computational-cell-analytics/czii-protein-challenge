import os
from typing import Optional, Tuple, Union

import numpy as np
from detection.data_processing.create_heatmap import get_label, parse_json_files

# TODO figure out how to correctly handle spotiflow dependency
try:
    from external.spotiflow.spotiflow.utils.peaks import points_to_flow3d
except ImportError:
    points_to_flow3d = None


def compute_stereographic_flow(
    coords: np.ndarray,
    patch_shape: Tuple[int, int, int],
    bb: Optional[Tuple[slice, ...]] = None,
    sigma: float = 1.5,
    grid: Union[int, Tuple[int, int, int]] = (1, 1, 1),
) -> np.ndarray:
    """
    Compute stereographic flow for a patch.

    Args:
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

    Returns:
    flow : np.ndarray
        Array with shape (4, Z, Y, X) and dtype float32 where the channel order is
        [w', z', y', x'] (same as spotiflow).
    """
    assert points_to_flow3d is not None, "Requires spotiflow dependency"

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
    # transpose and return (4, Z, Y, X) as float32
    flow_np = np.transpose(flow_np, (3, 0, 1, 2)).astype(np.float32, copy=False)
    return flow_np


class HeatmapTransform:
    """
    Returns heatmap label
    """
    def __init__(self, eps, sigma, lower_bound, upper_bound):
        self.eps = eps
        self.sigma = sigma
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __call__(self, label_path, shape, bb_labels):
        heatmap = get_label(
            label_path,
            shape,
            eps=self.eps,
            sigma=self.sigma,
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            bb=bb_labels,
        )
        return np.asarray(heatmap, dtype=np.float32)


class FlowTransform:
    """
    Returns stereographic flow label. 
    Output channel needs to be image dim + 1. 
    This Transform is only implemented for 3D.
    """
    def __init__(self, sigma=1.5, grid=(1, 1, 1)):
        self.sigma = sigma
        self.grid = grid

    def __call__(self, label_path, patch_spatial_shape, bb_for_loading):
        picks_folder = os.path.join(label_path, "Picks")

        json_files = [
            os.path.join(picks_folder, f)
            for f in os.listdir(picks_folder)
            if f.endswith(".json")
        ]

        coords, _ = parse_json_files(json_files)

        flow = compute_stereographic_flow(
            coords,
            patch_spatial_shape,
            bb=bb_for_loading,
            sigma=self.sigma,
            grid=self.grid,
        )

        return flow


class HeatmapFlowTransform:
    """
    Returns combined labels (heatmap and stereographic flow) of shape (1 + 4, Z, Y, X)
    """

    def __init__(
        self,
        eps,
        sigma,
        lower_bound,
        upper_bound,
        flow_sigma=None,
        flow_grid=(1, 1, 1),
    ):
        self.heatmap_transform = HeatmapTransform(
            eps, sigma, lower_bound, upper_bound
        )

        self.flow_transform = FlowTransform(
            sigma=flow_sigma if flow_sigma is not None else sigma,
            grid=flow_grid,
        )

    def __call__(self, label_path, shape, bb_labels, bb_for_loading):
        # Heatmap
        heatmap = self.heatmap_transform(
            label_path, shape, bb_labels
        )

        patch_spatial_shape = heatmap.shape[-3:]

        # stereographic Flow
        flow = self.flow_transform(
            label_path,
            patch_spatial_shape,
            bb_for_loading,
        )

        # Add heatmap channel
        heatmap_channel = heatmap[np.newaxis, ...].astype(np.float32, copy=False)

        # Combine
        labels = np.concatenate([heatmap_channel, flow], axis=0) # (1+4, Z, Y, X)

        return labels