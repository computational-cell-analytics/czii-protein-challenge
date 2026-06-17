import torch


def get_default_tiling():
    """Determine the tile shape and halo depending on the available VRAM.
    """
    if torch.cuda.is_available():
        print("Determining suitable tiling")

        # We always use the same default halo.
        halo = {"x": 64, "y": 64, "z": 32}  # before 64,64,16 (z raised to >=32)

        # Determine the GPU RAM and derive a suitable tiling.
        # NOTE: the network input along each axis equals `tile` (see prediction.py:
        # block_shape = tile - 2*halo, input = block_shape + 2*halo = tile), so `tile`
        # must stay divisible by the U-Net factor (z%8, x/y%16) and `tile - 2*halo` (the
        # block_shape / output step) must stay positive. With halo z=32 we raise each tile
        # z by 32 vs. before to keep the same block_shape z (80->48, 64->32, 48->16).
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9

        if vram >= 80:
            tile = {"x": 640, "y": 640, "z": 112}
        elif vram >= 40:
            tile = {"x": 512, "y": 512, "z": 96}
        elif vram >= 20:
            tile = {"x": 352, "y": 352, "z": 80}
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
