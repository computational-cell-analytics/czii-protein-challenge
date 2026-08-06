"""napari reader for the volume formats used in this project: HDF5, MRC/MRCS, TIFF and Zarr."""

import os
import warnings

import numpy as np

H5_EXTENSIONS = (".h5", ".hdf5", ".hdf")
MRC_EXTENSIONS = (".mrc", ".mrcs", ".rec", ".map", ".st", ".ali")
TIF_EXTENSIONS = (".tif", ".tiff")
ZARR_EXTENSIONS = (".zarr",)

ALL_EXTENSIONS = H5_EXTENSIONS + MRC_EXTENSIONS + TIF_EXTENSIONS + ZARR_EXTENSIONS

LABEL_KEYWORDS = ("label", "segmentation", "seg", "mask", "annotation", "instance")
POINT_KEYWORDS = ("coord", "point", "pick", "center", "centre", "position")

# Arrays below this size are read into RAM; larger ones stay lazy (dask over the file).
EAGER_LIMIT = 256 * 1024**2

# napari only keeps the arrays, so the h5py/mrcfile objects backing lazy reads
# would be garbage collected (and closed) mid-session without this.
_OPEN_FILES = []


def napari_get_reader(path):
    """Entry point: return a reader callable if we recognise `path`, else None."""
    paths = [path] if isinstance(path, str) else list(path)
    if not paths or not all(_is_supported(p) for p in paths):
        return None
    return read_paths


def read_paths(path):
    paths = [path] if isinstance(path, str) else list(path)
    layers = []
    for p in paths:
        layers.extend(_read_one(p))
    return layers or [(None,)]


def _is_supported(path):
    if not isinstance(path, str):
        return False
    ext = _extension(path)
    if ext in ZARR_EXTENSIONS:
        return True
    return ext in ALL_EXTENSIONS and os.path.isfile(path)


def _extension(path):
    return os.path.splitext(path.rstrip("/"))[1].lower()


def _read_one(path):
    ext = _extension(path)
    if ext in H5_EXTENSIONS:
        return _read_hdf5(path)
    if ext in MRC_EXTENSIONS:
        return _read_mrc(path)
    if ext in TIF_EXTENSIONS:
        return _read_tiff(path)
    if ext in ZARR_EXTENSIONS:
        return _read_zarr(path)
    raise ValueError(f"Unsupported file format: {path}")


#
# format-specific readers
#


def _read_hdf5(path):
    import h5py

    f = h5py.File(path, "r")
    _OPEN_FILES.append(f)

    stem = _stem(path)
    entries = []
    f.visititems(lambda name, obj: entries.append((name, obj)) if isinstance(obj, h5py.Dataset) else None)

    layers = []
    for name, dset in entries:
        if dset.ndim < 2 or 0 in dset.shape:
            continue
        layer = _dataset_to_layer(dset, f"{stem}/{name}", _scale_from_h5(dset, f))
        if layer is not None:
            layers.append(layer)

    if not layers:
        f.close()
        _OPEN_FILES.remove(f)
        raise ValueError(f"No displayable datasets found in {path}")
    return _arrange(layers)


def _read_mrc(path):
    import mrcfile

    with warnings.catch_warnings():  # headers written by RELION/IMOD are often non-standard
        warnings.simplefilter("ignore")
        mrc = mrcfile.mmap(path, mode="r", permissive=True)
    _OPEN_FILES.append(mrc)

    data = mrc.data
    scale = _scale_from_mrc(mrc, data.ndim)
    name = _stem(path)
    if _looks_like_labels(name, data.dtype):
        return [(data, {"name": name, "scale": scale}, "labels")]
    return [(data, _image_kwargs(name, scale), "image")]


def _read_tiff(path):
    import tifffile

    try:
        data = tifffile.memmap(path, mode="r")
    except (ValueError, MemoryError):  # compressed or non-contiguous
        data = tifffile.imread(path)

    name = _stem(path)
    scale = _scale_from_tiff(path, data.ndim)
    if _looks_like_labels(name, data.dtype):
        return [(data, {"name": name, "scale": scale}, "labels")]
    return [(data, _image_kwargs(name, scale), "image")]


def _read_zarr(path):
    import zarr

    root = zarr.open(path, mode="r")
    stem = _stem(path)

    if isinstance(root, zarr.core.Array):
        return [_array_to_layer(root, stem, None)]

    layers = []

    def visit(group, prefix):
        # Multiscale pyramid (copick / OME-Zarr): only take the full-resolution level.
        if "0" in group and isinstance(group["0"], zarr.core.Array):
            layers.append(_array_to_layer(group["0"], f"{stem}/{prefix}".rstrip("/"), None))
            return
        for key, item in group.items():
            sub = f"{prefix}/{key}".strip("/")
            if isinstance(item, zarr.core.Array):
                if item.ndim >= 2 and 0 not in item.shape:
                    layers.append(_array_to_layer(item, f"{stem}/{sub}", None))
            else:
                visit(item, sub)

    visit(root, "")
    if not layers:
        raise ValueError(f"No displayable arrays found in {path}")
    return _arrange(layers)


#
# helpers
#


def _dataset_to_layer(dset, name, scale):
    """Turn an h5py dataset into a napari layer tuple, or None if it is not displayable."""
    if _looks_like_points(name, dset):
        return (np.asarray(dset), {"name": name, "size": 10, "out_of_slice_display": True}, "points")
    return _array_to_layer(dset, name, scale)


def _array_to_layer(source, name, scale):
    data = _lazy(source)
    if _looks_like_labels(name, data.dtype):
        return (data, {"name": name, "scale": scale}, "labels")
    return (data, _image_kwargs(name, scale), "image")


def _lazy(source):
    """Read small arrays eagerly, wrap large ones in dask so we never load a whole tomogram."""
    nbytes = source.size * source.dtype.itemsize
    if nbytes <= EAGER_LIMIT:
        return np.asarray(source)
    import dask.array as da

    chunks = getattr(source, "chunks", None) or "auto"
    return da.from_array(source, chunks=chunks)


def _image_kwargs(name, scale):
    return {"name": name, "scale": scale, "rendering": "mip", "blending": "translucent"}


def _arrange(layers):
    """A file often holds several datasets; stacking every image on top of each other is unreadable.

    So: put the most likely "main" volume first and hide the other images. Labels and points are
    overlays, so they stay visible.
    """
    images = sorted([l for l in layers if l[2] == "image"], key=_image_rank)
    overlays = [l for l in layers if l[2] != "image"]
    images = images[:1] + [(d, {**meta, "visible": False}, t) for d, meta, t in images[1:]]
    return images + overlays


def _image_rank(layer):
    """Prefer the raw/denoised tomogram, then the biggest array."""
    data, meta, _ = layer
    name = meta["name"].lower()
    preferred = any(k in name for k in ("raw", "denoised", "image", "data", "tomogram", "/0"))
    return (not preferred, -int(np.prod(data.shape)))


def _looks_like_labels(name, dtype):
    lowered = name.lower()
    return np.issubdtype(dtype, np.integer) and any(k in lowered for k in LABEL_KEYWORDS)


def _looks_like_points(name, dset):
    lowered = name.lower()
    return dset.ndim == 2 and dset.shape[1] in (2, 3) and any(k in lowered for k in POINT_KEYWORDS)


def _stem(path):
    return os.path.basename(path.rstrip("/"))


def _scale_from_mrc(mrc, ndim):
    try:
        vs = mrc.voxel_size
        scale = [float(vs.z), float(vs.y), float(vs.x)]
    except Exception:
        return None
    if not all(np.isfinite(s) and s > 0 for s in scale):
        return None
    return scale[-ndim:] if ndim <= 3 else [1.0] * (ndim - 3) + scale


def _scale_from_h5(dset, f):
    """Pick up the voxel size conventions written by elf / ilastik, if present."""
    for source in (dset.attrs, f.attrs):
        for key in ("element_size_um", "voxel_size", "pixel_size", "resolution"):
            if key not in source:
                continue
            value = np.atleast_1d(np.asarray(source[key], dtype=float))
            if value.size == dset.ndim and np.all(np.isfinite(value)) and np.all(value > 0):
                return list(value)
    return None


def _scale_from_tiff(path, ndim):
    import tifffile

    try:
        with tifffile.TiffFile(path) as tif:
            meta = tif.imagej_metadata or {}
            page = tif.pages[0]
            xres = page.tags.get("XResolution")
            if xres is None:
                return None
            num, den = xres.value
            xy = den / num if num else 1.0
            z = float(meta.get("spacing", xy))
    except Exception:
        return None
    if not (np.isfinite(xy) and xy > 0):
        return None
    scale = [z, xy, xy][-min(ndim, 3):]
    return [1.0] * (ndim - len(scale)) + scale
