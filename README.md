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

## Repository Structure

- `detection/` – 3D U-Net density prediction and optional stereographic flow
- `classification/` – 3D ResNet-based protein classification
- `external/` – external code or resources

THIS IS WORK IN PROGRESS!

## Installation

- Make sure conda or mamba is installed.
    - If you don't have a conda installation yet we recommend [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)
- Create the environment with all required dependencies: `mamba env create -f environment.yaml`
- Activate the environment: `mamba activate pro-revelio`
- Install the package: `pip install -e .`

