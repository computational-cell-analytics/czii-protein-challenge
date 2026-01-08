import torch


def get_default_tiling():
    """Determine the tile shape and halo depending on the available VRAM.
    """
    if torch.cuda.is_available():
        print("Determining suitable tiling")

        # We always use the same default halo.
        halo = {"x": 64, "y": 64, "z": 16}  # before 64,64,8

        # Determine the GPU RAM and derive a suitable tiling.
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9

        if vram >= 80:
            tile = {"x": 640, "y": 640, "z": 80}
        elif vram >= 40:
            tile = {"x": 512, "y": 512, "z": 64}
        elif vram >= 20:
            tile = {"x": 352, "y": 352, "z": 48}
        else:
            # TODO determine tilings for smaller VRAM
            raise NotImplementedError(f"Estimating the tile size for a GPU with {vram} GB is not yet supported.")

        print(f"Determined tile size: {tile}")
        tiling = {"tile": tile, "halo": halo}

    # I am not sure what is reasonable on a cpu. For now choosing very small tiling.
    # (This will not work well on a CPU in any case.)
    else:
        print("Determining default tiling")
        tiling = {
            "tile": {"x": 96, "y": 96, "z": 16},
            "halo": {"x": 16, "y": 16, "z": 4},
        }

    return tiling


def parse_tiling(tile_shape, halo):
    """
    Helper function to parse tiling parameter input from the command line.

    Args:
        tile_shape: The tile shape. If None the default tile shape is used.
        halo: The halo. If None the default halo is used.

    Returns:
        dict: the tiling specification
    """

    default_tiling = get_default_tiling()

    if tile_shape is None:
        tile_shape = default_tiling["tile"]
    else:
        assert len(tile_shape) == 3
        tile_shape = dict(zip("zyx", tile_shape))

    if halo is None:
        halo = default_tiling["halo"]
    else:
        assert len(halo) == 3
        halo = dict(zip("zyx", halo))

    tiling = {"tile": tile_shape, "halo": halo}
    return tiling
