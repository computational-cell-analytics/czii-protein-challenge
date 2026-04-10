import os
import time
import warnings
from typing import Tuple

import torch
import numpy as np
import torch_em
from torch_em.model.resnet3d import resnet3d_18
import torch.nn as nn
from classification.utils.training import CryoETNormalize


def pad_to_patch(subtomograms: np.ndarray, patch_shape=(64, 64, 64)):
    """
    Pad subtomograms so they have at least the given patch_shape.
    
    Args:
        subtomograms (np.ndarray): shape (N, D, H, W) or (D, H, W)
        patch_shape (tuple): target minimal shape (D, H, W)
    
    Returns:
        np.ndarray: padded subtomograms with shape (N, D', H', W'),
                    where each dim D' >= patch_shape[0], etc.
    """
    # Ensure batch dimension
    if subtomograms.ndim == 3:
        subtomograms = np.expand_dims(subtomograms, axis=0)

    N, D, H, W = subtomograms.shape
    target_D, target_H, target_W = patch_shape

    # Compute padding for each dimension
    pad_d = max(0, target_D - D)
    pad_h = max(0, target_H - H)
    pad_w = max(0, target_W - W)

    # Split padding equally left/right (extra goes to the right)
    pad_before_d, pad_after_d = pad_d // 2, pad_d - pad_d // 2
    pad_before_h, pad_after_h = pad_h // 2, pad_h - pad_h // 2
    pad_before_w, pad_after_w = pad_w // 2, pad_w - pad_w // 2

    padding = (
        (0, 0),  # batch dim, no padding
        (pad_before_d, pad_after_d),
        (pad_before_h, pad_after_h),
        (pad_before_w, pad_after_w)
    )

    # Apply padding (0 padding atm, #TODO do a different one?)
    subtomograms_padded = np.pad(subtomograms, padding, mode="constant", constant_values=0)

    return subtomograms_padded


def get_model(model_path, device, EfficientNet=False):
    if EfficientNet:
        from external.efficientnet3d.efficientnet_pytorch_3d import EfficientNet3D
        model = EfficientNet3D.from_name("efficientnet-b0", override_params={'num_classes': 7}, in_channels=1)
    else:
        model = resnet3d_18(
            in_channels=1,
            out_channels=7
        )
        
        # Replace conv1 and maxpool
        model.conv1 = nn.Conv3d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()  # remove pooling to preserve resolution

    model_path = os.path.join(model_path, "best.pt")
    checkpoint = torch.load(model_path, map_location=device, weights_only = False)
    model.load_state_dict(checkpoint['model_state'], strict=False)
    model.eval()
    model.to(device)

    return model, checkpoint


def protein_classification(
    subtomograms: np.ndarray,  # [z, y, x] or (N, z, y, x)
    model: torch.nn.Module = None,
    model_path: str = None,
    verbose: bool = True,
    device: str = None,
    EfficientNet: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classify one or more subtomograms.
    
    Args:
        subtomograms (np.ndarray): shape (N, D, H, W) or (D, H, W) for a single cube
        model (torch.nn.Module, optional): Preloaded model. If given, model_path is ignored.
        model_path (str): path to model file or directory
        verbose (bool): Whether to print timing information
        device (str or torch.device): 'cpu' or 'cuda' or torch.device object, defaults to available device
    
    Returns:
        probs (np.ndarray): shape (N, num_classes) probabilities per class
        preds (np.ndarray): shape (N,) predicted class indices
    """
    if verbose:
        print("Predicting protein location in volume of shape", subtomograms.shape)

    if subtomograms.ndim == 3:  # single cube
        subtomograms = np.expand_dims(subtomograms, axis=0)

    #if EfficientNet:
        #pad is the subtomograms dimensions are too small (<64x64x64) 
        #subtomograms = pad_to_patch(subtomograms)

    #normalise
    #from torch_em.transform.raw import normalize
    #subtomograms = np.stack([normalize(st) for st in subtomograms])
    normalizer = CryoETNormalize()
    subtomograms = torch.stack(
        [normalizer(st) for st in subtomograms]
    ).numpy()


    t0 = time.time()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    if model is None:
        if model_path is None:
            raise ValueError("Either 'model' or 'model_path' must be provided.")

        if model_path.endswith("best.pt"):
            model_path = os.path.split(model_path)[0]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model, _ = get_model(
                model_path=model_path,
                device=device,
                EfficientNet=EfficientNet
            )
    else:
        model = model.to(device)
        model.eval()

    with torch.no_grad():
        tensor = torch.from_numpy(subtomograms).float().unsqueeze(1).to(device)  # (N, 1, D, H, W)
        logits = model(tensor)  # (N, num_classes)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

    if verbose:
        print("Prediction time:", time.time() - t0, "s")

    return probs, preds
