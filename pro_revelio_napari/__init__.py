"""napari plugin for viewing and comparing cryo-ET volumes (HDF5, MRC/MRCS, TIFF, Zarr) in 3D.

Nothing is imported here on purpose: npe2 resolves the contributions by their full module path, and
an eager import would pull Qt in on plugin discovery.
"""
