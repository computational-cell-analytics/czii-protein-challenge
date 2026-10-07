from .gridsearch import gridsearch
from .postprocessing import detections_from_prediction


def protein_detection(heatmap, json_val_path, model_path, threshold=None):
    """
    Detect protein coordinates from a heatmap and adjust them using stereographic flow predictions.

    Args:
        heatmap (np.ndarray): Prediction with shape (5, D, H, W), see `detections_from_prediction`.
        json_val_path (str): Path to validation JSON for threshold optimization.
        model_path (str): Path to model for gridsearch.
        threshold (float, optional): Detection threshold. If None, determined via gridsearch.

    Returns:
        pred_coords (list): Adjusted coordinates [[z, y, x], ...].
        threshold (float): Used detection threshold.
    """

    if threshold is None:
        threshold = gridsearch(json_val_path, model_path)

    coords, _ = detections_from_prediction(heatmap, threshold)

    return coords.tolist(), threshold
