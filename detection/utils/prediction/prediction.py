import os
import time
import warnings
# from glob import glob
from typing import Dict  # , Optional, Tuple

import numpy as np
import torch
import torch_em

from torch_em.util.prediction import predict_with_halo


def get_prediction_torch_em(
    input_volume: np.ndarray,  # [z, y, x]
    tiling: Dict[str, Dict[str, int]],  # {"tile": {"z": int, ...}, "halo": {"z": int, ...}}
    model_path: str = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    Run prediction using torch-em on a given volume.

    Args:
        input_volume: The input volume to predict on.
        model_path: The path to the model checkpoint if 'model' is not provided.
        tiling: The tiling configuration for the prediction.
        verbose: Whether to print timing information.

    Returns:
        The predicted volume.
    """
    if verbose:
        print("Predicting protein location in volume of shape", input_volume.shape)

    if model_path.endswith("best.pt"):
        model_path = os.path.split(model_path)[0]

    print(f"tiling {tiling}")
    # Create updated_tiling with the same structure
    updated_tiling = {
        "tile": {},
        "halo": tiling["halo"]  # Keep the halo part unchanged
    }
    # Update tile dimensions
    for dim in tiling["tile"]:
        updated_tiling["tile"][dim] = tiling["tile"][dim] - 2 * tiling["halo"][dim]
    print(f"updated_tiling {updated_tiling}")

    # get block_shape and halo
    block_shape = [updated_tiling["tile"]["z"], updated_tiling["tile"]["x"], updated_tiling["tile"]["y"]]
    halo = [updated_tiling["halo"]["z"], updated_tiling["halo"]["x"], updated_tiling["halo"]["y"]]

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Suppress warning when loading the model.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if os.path.isdir(model_path):  # Load the model from a torch_em checkpoint.
            model = torch_em.util.load_model(checkpoint=model_path, device=device)
        else:  # Load the model directly from a serialized pytorch model.
            model = torch.load(model_path)

    # Run prediction with the model.
    with torch.no_grad():

        pred = predict_with_halo(
            input_volume, model, gpu_ids=[device],
            block_shape=block_shape, halo=halo,
            preprocess=None,
        )
    if verbose:
        print("Prediction time in", time.time() - t0, "s")

    return pred
