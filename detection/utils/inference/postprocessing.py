import numpy as np
from skimage.feature import peak_local_max

from detection.config import ADJ_FACTOR, FLOW_SIGMA, CZII_SMALLEST_PROTEIN_SIZE

# Minimum separation between peaks, and the NMS radius applied after the flow correction has
# pulled neighbouring peaks together. Both are derived from the smallest protein.
PEAK_MIN_DISTANCE = int(CZII_SMALLEST_PROTEIN_SIZE * ADJ_FACTOR)
NMS_RADIUS = CZII_SMALLEST_PROTEIN_SIZE * ADJ_FACTOR * 0.9


def _nms(coords, scores, radius):
    """Greedy NMS: keep the highest-scoring prediction; suppress all others within radius."""
    order = np.argsort(scores)[::-1]
    kept = []
    suppressed = set()
    for i in order:
        if i in suppressed:
            continue
        kept.append(i)
        for j in order:
            if j != i and j not in suppressed:
                if np.linalg.norm(coords[i] - coords[j]) < radius:
                    suppressed.add(j)
    return kept


def detections_from_prediction(prediction, threshold):
    """
    Turn a raw network prediction into detected coordinates.

    This is the whole post-processing chain - peak finding, stereographic flow correction and
    NMS - so that threshold selection (gridsearch) and inference score the same coordinates.

    Args:
        prediction (np.ndarray): Tensor with shape (5, D, H, W) containing:
                                 [0] heatmap,
                                 [1] w' (stereographic scale),
                                 [2] z' (flow z),
                                 [3] y' (flow y),
                                 [4] x' (flow x)
        threshold (float): Detection threshold on the heatmap.

    Returns:
        coords (np.ndarray): Adjusted coordinates, shape (N, 3) as [z, y, x].
        scores (np.ndarray): Heatmap value at the peak each coordinate came from, shape (N,).
    """
    heatmap = prediction[0]

    # Find peaks in heatmap. skimage keeps peaks strictly above threshold_abs.
    pred_coords = peak_local_max(
        heatmap,
        min_distance=PEAK_MIN_DISTANCE,
        threshold_abs=threshold,
    )

    if len(pred_coords) == 0:
        return np.zeros((0, 3)), np.zeros(0)

    # Extract flow channels
    flow_w = prediction[1]
    flow_z = prediction[2]
    flow_y = prediction[3]
    flow_x = prediction[4]

    s = FLOW_SIGMA

    # Apply stereographic flow correction and clip to volume bounds
    D, H, W = heatmap.shape
    adjusted_coords = []
    scores = []
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

        adj_z = float(min(max(z + dz, 0), D - 1))
        adj_y = float(min(max(y + dy, 0), H - 1))
        adj_x = float(min(max(x + dx, 0), W - 1))

        adjusted_coords.append([adj_z, adj_y, adj_x])
        scores.append(float(heatmap[z, y, x]))

    adjusted_coords = np.array(adjusted_coords)
    scores = np.array(scores)

    # NMS: suppress duplicate peaks that converged to the same location after flow correction
    kept = _nms(adjusted_coords, scores, NMS_RADIUS)
    return adjusted_coords[kept], scores[kept]
