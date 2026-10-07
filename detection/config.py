#TODO fill with more global variables

# this factor is used to translate the pdb protein sizes to match the protein sizes in the actual experimental data
ADJ_FACTOR = 0.3

# sigma used for the stereographic flow
FLOW_SIGMA = 1.5

# --- Threshold gridsearch ---
# F-beta weighting used to pick the detection threshold over the validation set.
# beta < 1 favours precision (fewer false positives / less oversampling); beta > 1
# favours recall. beta=0.5 weights precision ~4x vs recall in the harmonic mean;
# lower it (e.g. 0.25 -> ~16x) to push precision harder against oversampling.
GRIDSEARCH_BETA = 0.25

# Threshold sweep for the gridsearch as (start, stop, step) passed to np.arange.
# Measured on v43 (detection_scripts/evaluation/measure_peak_response.py): the response is
# bimodal - a particle either gets ~4 or gets background - so the boundary is where the
# background tail ends and the hits begin:
#     simulated val   background p99 2.935, hits p10 3.97 .. max 4.54
#     real public_test background p99 2.417, hits p50 3.33 .. max 4.63
# The old ceiling of 3.0 (np.arange stops at 2.9) sat just *below* that boundary, so the
# sweep could never select a threshold that separates hits from background. The floor is
# kept low deliberately: if the optimum lands on it again, something is wrong upstream.
GRIDSEARCH_THRESH_RANGE = (1.0, 4.3, 0.1)

#smallest protein structure for CZII dataset: "beta-amylase": 33.27
'''
info:
smalles protein structure: "beta-amylase": 33.27
bigges protein structure: "ribosome": 109.02
'''
CZII_SMALLEST_PROTEIN_SIZE = 33.27

# --- Matching tolerance ---
# Picks are stored in Angstrom; parse_json_files divides by this, so every coordinate
# downstream (labels, detections, matching) is in voxels of a VoxelSpacing10 tomogram.
VOXEL_SIZE_A = 10.0

# Official CZII particle radii in Angstrom.
CZII_PARTICLE_RADII_A = {
    "apo-ferritin": 60,
    "beta-amylase": 65,
    "beta-galactosidase": 90,
    "ribosome": 150,
    "thyroglobulin": 130,
    "virus-like-particle": 135,
}

# CZII counts a pick as found when the prediction is within this fraction of its radius.
CZII_MATCH_TOLERANCE_FACTOR = 0.5

# Match tolerance in voxels, per class: 3.0 (apo-ferritin) to 7.5 (ribosome).
CZII_MATCH_DISTANCE_VOXELS = {
    name: CZII_MATCH_TOLERANCE_FACTOR * radius / VOXEL_SIZE_A
    for name, radius in CZII_PARTICLE_RADII_A.items()
}

# Used for classes without a published radius (e.g. simulated-only structures). The mean
# of the six CZII tolerances, so an unknown class is neither favoured nor penalised.
DEFAULT_MATCH_DISTANCE = sum(CZII_MATCH_DISTANCE_VOXELS.values()) / len(CZII_MATCH_DISTANCE_VOXELS)


# --- Per-class heatmap loss weights ---
# The Gaussian peak is 3.989 for every class, so the loss a particle contributes scales with
# sigma**3, i.e. with its size: a VLP carries ~43x the gradient signal of a beta-amylase
# (and ~215x under the old width_dict, which truncated its sigma to 1). These weights are
# mean(sum(t^2)) / sum(t^2) per class under the current stamp geometry, so **every particle
# contributes equally to the heatmap loss regardless of its size** and a mid-sized class sits
# at ~1.0. Re-derive them if CZII_STAMP_WIDTH_VOXELS or the eps/bounds change.
#
# NB: this equalises per *instance*. Per *class* balance additionally needs comparable particle
# counts in the training set - see simulation/configs/czii_c7.toml.
CZII_CLASS_LOSS_WEIGHTS = {
    "apo-ferritin": 4.4352,
    "beta-amylase": 15.0048,
    "beta-galactosidase": 1.8833,
    "ribosome": 0.5558,
    "thyroglobulin": 1.8705,
    "virus-like-particle": 0.3517,
}

# Weight given to voxels that belong to no particle. 1.0 keeps the background contributing
# exactly as it does today, so these weights only rebalance the classes against each other.
BACKGROUND_LOSS_WEIGHT = 1.0
