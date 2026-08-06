# czii-protein-challenge

This repository contains tools and workflows for building machine learning models to identify and annotate protein complexes in 3D cellular images. By leveraging cryo-electron tomography (cryoET) data, the project aims to advance our ability to understand protein interactions in their native cellular environments.

The pipeline consists of two main components: protein detection and protein classification.


## 1. Protein Detection

The detection stage predicts candidate protein locations in 3D tomograms.

### Model Architecture

- A 3D U-Net is used to predict a volumetric density map.
- The target density map is created according to the STACC formulation.
- Optionally, a stereographic flow field can be predicted following the Spotiflow implementation.

### Detection Strategy

- Protein object centers are identified as local maxima in the predicted density map.
- If stereographic flow is enabled, the predicted vector field is used to refine object locations.

This stage outputs candidate 3D coordinates for protein complexes.


## 2. Protein Classification

The classification stage assigns a protein type to each detected location.

### Model Architecture

- A 3D ResNet processes cropped subvolumes centered at detected coordinates.

### Workflow

1. The detection model predicts candidate protein locations.
2. For each detected location, a 3D patch is extracted.
3. The 3D ResNet classifies the corresponding protein complex.

This modular setup allows detection and classification to be trained, evaluated, and improved independently.

---

## 3. Napari Viewer Plugin

`pro_revelio_napari/` is a napari plugin for looking at the volumes this pipeline produces.

- **Drag and drop** `.h5` / `.hdf5`, `.mrc` / `.mrcs` / `.rec` / `.map` / `.st` / `.ali`, `.tif` / `.tiff`
  and `.zarr` directories straight onto the napari window.
- Open the panel via **Plugins → Pro-Revelio → 3D Volume Viewer** for the 3D controls: rendering mode
  (mip, attenuated mip, iso, …), percentile auto-contrast, colormap, gamma, opacity, z stretch and
  camera rotation. Every control can target one layer or all image layers at once.
- The panel also has its own drop area and file/folder browser.

Details worth knowing:

- Volumes above 256 MB are loaded lazily (dask over the file), so opening a large tomogram is instant
  and only the displayed slices are read.
- HDF5 and Zarr files are walked recursively. All datasets are added, but only the most likely main
  volume is visible — the others are added hidden so they don't stack on top of each other.
  Multiscale Zarr pyramids (copick / OME-Zarr) load the full-resolution level only.
- Datasets whose name contains `label`/`seg`/`mask`/`annotation` become **Labels** layers, and
  `(N, 2)`/`(N, 3)` datasets whose name contains `coord`/`point`/`pick`/`center` become **Points**
  layers. Points are taken in the order they are stored — napari expects `(z, y, x)`, so an array
  saved as `xyz` will need flipping.
- Voxel sizes are read from MRC headers, from the `element_size_um` / `voxel_size` / `pixel_size`
  HDF5 attributes, and from ImageJ TIFF metadata.
- On first use the plugin registers itself as napari's preferred reader for these extensions, so
  drops don't ask which plugin to use. Untick *"Use this plugin for drag & drop"* in the panel to
  hand them back to the napari builtins.

## Repository Structure

- `detection/` – 3D U-Net density prediction and optional stereographic flow
- `classification/` – 3D ResNet-based protein classification
- `pro_revelio_napari/` – napari reader + 3D viewer plugin for h5/mrc/tif/zarr volumes
- `detection_scripts/` - example scirpts for training a detection model and running protein coordinates prediction
- `detection_scripts/evaluation/` - example scirpts for evaluating predicted protein coordinates
- `classification_scripts/` - example scripts for training a classification model and running protein prediction and evaluation on either GT or predicted coordinates
- `external/` – external code or resources; if EfficientNet is not needed, this can be deleted

THIS IS WORK IN PROGRESS!

## Installation

- Make sure conda or mamba is installed.
    - If you don't have a conda installation yet we recommend [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)
- Create the environment with all required dependencies: `mamba env create -f environment.yaml`
- Activate the environment: `mamba activate pro-revelio`
- Install the package: `pip install -e .`

