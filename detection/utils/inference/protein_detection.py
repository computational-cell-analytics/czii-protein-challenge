from skimage.feature import peak_local_max
from .gridsearch import gridsearch

from detection.config import ADJ_FACTOR, FLOW_SIGMA, CZII_SMALLEST_PROTEIN_SIZE


def protein_detection(heatmap, json_val_path, model_path, threshold=None):
    """
    Detect protein coordinates from a heatmap and adjust them using stereographic flow predictions.

    Args:
        heatmap (np.ndarray): Tensor with shape (5, D, H, W) containing:
                              [0] heatmap,
                              [1] w' (stereographic scale),
                              [2] z' (flow z),
                              [3] y' (flow y),
                              [4] x' (flow x)
        json_val_path (str): Path to validation JSON for threshold optimization.
        model_path (str): Path to model for gridsearch.
        threshold (float, optional): Detection threshold. If None, determined via gridsearch.

    Returns:
        pred_coords (list): Adjusted coordinates [[z, y, x], ...].
        threshold (float): Used detection threshold.
    """

    if threshold is None:
        threshold = gridsearch(json_val_path, model_path) 
    
    adj_factor = ADJ_FACTOR

    # Find peaks in heatmap
    pred_coords = peak_local_max(
        heatmap[0],
        min_distance=int(CZII_SMALLEST_PROTEIN_SIZE * adj_factor * 0.9),
        threshold_abs=threshold
    )

    # Apply stereographic flow correction
    # Extract flow channels
    flow_w = heatmap[1]
    flow_z = heatmap[2]
    flow_y = heatmap[3]
    flow_x = heatmap[4]

    s = FLOW_SIGMA

    # Adjust coordinates using predicted local flow
    adjusted_coords = []
    for z, y, x in pred_coords:
        w = flow_w[z, y, x]
        vz_ = flow_z[z, y, x]
        vy_ = flow_y[z, y, x]
        vx_ = flow_x[z, y, x]

        # like Spotiflow: using scaled stereographic projection with parameter s (sigma from training) to adjust coordinates
        # can do it like this since my model is predicting the normalised flow calculated by the points_to_flow3d function from spotiflow
        denom = 1.0 + w + 1e-8  # avoid division by zero

        dz = s * vz_ / denom
        dy = s * vy_ / denom
        dx = s * vx_ / denom

        adj_z = z + dz
        adj_y = y + dy
        adj_x = x + dx

        adjusted_coords.append([float(adj_z), float(adj_y), float(adj_x)])

    return adjusted_coords, threshold