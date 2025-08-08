import os
import time
import warnings
from typing import Tuple

import torch
import numpy as np
import torch_em

def protein_classification(
    subtomograms: np.ndarray,  # [z, y, x] or (N, z, y, x)
    model_path: str = None,
    verbose: bool = True,
    device: str = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classify one or more subtomograms.
    
    Args:
        subtomograms (np.ndarray): shape (N, D, H, W) or (D, H, W) for a single cube
        model_path (str): path to model file or directory
        verbose (bool): Whether to print timing information
        device (str or torch.device): 'cpu' or 'cuda' or torch.device object, defaults to available device
    
    Returns:
        probs (np.ndarray): shape (N, num_classes) probabilities per class
        preds (np.ndarray): shape (N,) predicted class indices
    """
    if verbose:
        print("Predicting protein location in volume of shape", subtomograms.shape)

    if model_path.endswith("best.pt"):
        model_path = os.path.split(model_path)[0]

    if subtomograms.ndim == 3:  # single cube
        subtomograms = np.expand_dims(subtomograms, axis=0)

    t0 = time.time()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if os.path.isdir(model_path):  # Load model from torch_em checkpoint dir
            model = torch_em.util.load_model(checkpoint=model_path, device=device)
        else:  # Load model directly from serialized pytorch model
            model = torch.load(model_path, map_location=device)

    model.eval()

    with torch.no_grad():
        tensor = torch.from_numpy(subtomograms).float().unsqueeze(1).to(device)  # (N, 1, D, H, W)
        logits = model(tensor)  # (N, num_classes)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

    if verbose:
        print("Prediction time:", time.time() - t0, "s")

    return probs, preds
