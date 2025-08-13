import numpy as np
from skimage.feature import blob_log, peak_local_max
from .gridsearch import gridsearch

def protein_detection(heatmap, json_val_path, model_path): #TODO do this properly
    """
    Detects local maxima and estimates sizes of Gaussians in a 3D heatmap.

    Parameters:
        heatmap (np.array): A 3D numpy array representing the heatmap.

    Returns:
        list of dict: A list of dictionaries with keys:
            - 'coordinates': Tuple of (z, y, x) for the local maxima
            - 'size': Estimated size of the Gaussian (sigma equivalent)
    """

    threshold = gridsearch(json_val_path, model_path) 
    #smalles protein structure: "beta-amylase": 33.27
    #bigges protein structure: "ribosome": 109.02,
    #0.3 is the factor to match the PDB size to the experimental data size
    adj_factor=0.3 #TODO implement this as an argument, also when creating heatmap
    #TODO decide on blob_log or peak_local_max; blob_log is SUPER slow
    '''pred_coords = blob_log(heatmap, min_sigma=33.27*adj_factor, max_sigma=109.02*adj_factor, threshold=threshold) 
    pred_coords = pred_coords[:, 1:-1]'''

    pred_coords = peak_local_max(heatmap[0], min_distance=int(33.27*adj_factor *0.9), threshold_abs=threshold)


    #TODO calculate size of each gaussians and save it in detections
    '''detections.append({
        'coordinates': pred_coords,
        'size': sizes
    })
'''
    return pred_coords.tolist()