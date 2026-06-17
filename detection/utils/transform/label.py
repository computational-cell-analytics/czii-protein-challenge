import os
from typing import Optional, Tuple, Union

import numpy as np
from detection.data_processing.create_heatmap import get_label, parse_json_files
from spotiflow.utils.peaks import points_to_flow3d


def _bb_starts_stops(bb: Tuple[slice, ...]) -> Tuple[np.ndarray, np.ndarray]:
    """Extract the spatial (z, y, x) starts and stops from a bounding box of slices.

    Drops a leading channel slice if present (bb may be (slice(None), z, y, x)).
    """
    starts = []
    stops = []
    for s in bb:
        if isinstance(s, slice):
            start = 0 if s.start is None else s.start
            stop = None if s.stop is None else s.stop
            starts.append(start)
            stops.append(stop)
    if len(starts) >= 3:
        # take last 3 slices if an extra channel slice was prepended
        starts = starts[-3:]
        stops = stops[-3:]
    elif len(starts) != 3:
        raise ValueError("bb must contain 3 spatial slices (z,y,x) or channel plus 3 slices.")
    return np.array(starts, dtype=np.float32), np.array(stops, dtype=np.float32)


def compute_stereographic_flow(
    coords: np.ndarray,
    patch_shape: Tuple[int, int, int],
    bb: Optional[Tuple[slice, ...]] = None,
    filter_bb: Optional[Tuple[slice, ...]] = None,
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
        If provided, the bounding-box (in full-image coordinates) of the loaded patch.
        Used to offset coords into patch-local coordinates by subtracting bb.start values,
        so the flow grid (of size `patch_shape`) is aligned with this bounding box.
    filter_bb : tuple(slice,...), optional
        If provided, only points whose absolute coordinates fall inside this bounding-box
        are used for the flow. Use this to restrict the flow to proteins inside the inner
        patch while still computing the flow on the (larger, halo-extended) `bb` grid.
        If None, points inside `bb` are used (the previous behaviour).
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
        starts, stops = _bb_starts_stops(bb)

        # Shift coords into patch-local coordinates (aligned with the `bb` grid)
        if coords.shape[0] > 0:
            coords_local = coords - starts[np.newaxis, :]
            if filter_bb is not None:
                # Keep only points inside the inner patch (absolute coordinates), so proteins
                # in the halo region just outside the patch are excluded from the flow.
                f_starts, f_stops = _bb_starts_stops(filter_bb)
                inside_mask = np.all(
                    (coords >= f_starts[np.newaxis, :]) & (coords < f_stops[np.newaxis, :]), axis=1
                )
            else:
                # Keep points that fall inside the loaded patch bounds [0, size)
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

    def __call__(self, label_path, shape, bb_labels, bb_for_loading=None):
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

    def __call__(self, label_path, shape, bb_for_loading=None, filter_bb=None):
        patch_spatial_shape = shape[-3:]

        json_files = [
            os.path.join(root, f)
            for root, _, files in os.walk(label_path)
            for f in files
            if f.endswith(".json") and f not in ("no_class.json", "albumin.json", "actin.json", "mt.json")
        ]

        coords, _ = parse_json_files(json_files)

        flow = compute_stereographic_flow(
            coords,
            patch_spatial_shape,
            bb=bb_for_loading,
            filter_bb=filter_bb,
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

    def __call__(self, label_path, shape, bb_labels, bb_for_loading, bb_inner=None):
        # Heatmap: computed over the (halo-extended) bb_labels, so proteins just outside
        # the patch (within the halo) still contribute their Gaussian tails.
        heatmap = self.heatmap_transform(
            label_path, shape, bb_labels
        )

        patch_spatial_shape = heatmap.shape[-3:]

        # stereographic Flow: computed on the same (halo-extended) grid as the heatmap
        # (via bb_for_loading), but restricted to proteins inside the inner patch
        # (bb_inner) so that points just outside the patch are NOT used for the flow.
        flow = self.flow_transform(
            label_path,
            patch_spatial_shape,
            bb_for_loading,
            bb_inner,
        )

        # Add heatmap channel
        heatmap_channel = heatmap[np.newaxis, ...].astype(np.float32, copy=False)

        # Combine
        labels = np.concatenate([heatmap_channel, flow], axis=0) # (1+4, Z, Y, X)

        return labels