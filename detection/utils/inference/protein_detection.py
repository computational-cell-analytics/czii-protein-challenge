import numpy as np
from skimage.feature import blob_log, peak_local_max
from .gridsearch import gridsearch


def protein_detection(heatmap, json_val_path, model_path, threshold=None): #TODO do this properly
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
    #smalles protein structure: "beta-amylase": 33.27
    #bigges protein structure: "ribosome": 109.02,
    #0.3 is the factor to match the PDB size to the experimental data size
    adj_factor=0.3 #TODO implement this as an argument, also when creating heatmap
    #TODO decide on blob_log or peak_local_max; blob_log is SUPER slow
    '''pred_coords = blob_log(heatmap, min_sigma=33.27*adj_factor, max_sigma=109.02*adj_factor, threshold=threshold) 
    pred_coords = pred_coords[:, 1:-1]'''

    # Find peaks in heatmap
    pred_coords = peak_local_max(
        heatmap[0],
        min_distance=int(33.27 * adj_factor * 0.9),
        threshold_abs=threshold
    )

    # Apply stereographic flow correction
    # Extract flow channels
    flow_w = heatmap[1]  # stereographic scaling (TODO needed later?)
    flow_z = heatmap[2]
    flow_y = heatmap[3]
    flow_x = heatmap[4]

    # Adjust coordinates using predicted local flow
    adjusted_coords = []
    for z, y, x in pred_coords:
        dz = flow_z[z, y, x]
        dy = flow_y[z, y, x]
        dx = flow_x[z, y, x]

        # Optionally apply stereographic scaling (if relevant)
        # In Spotiflow, coordinates are typically adjusted directly by the flow values TODO
        adj_z = z + dz
        adj_y = y + dy
        adj_x = x + dx

        adjusted_coords.append([float(adj_z), float(adj_y), float(adj_x)])


    #TODO calculate size of each gaussians and save it in detections
    '''detections.append({
        'coordinates': pred_coords,
        'size': sizes
    })
'''
    return adjusted_coords, threshold