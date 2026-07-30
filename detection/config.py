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
# The upper bound is extended past the old 2.5 so a strong precision threshold is
# not clipped at the top of the range.
GRIDSEARCH_THRESH_RANGE = (1.0, 3.0, 0.1)

#smallest protein structure for CZII dataset: "beta-amylase": 33.27
'''
info:
smalles protein structure: "beta-amylase": 33.27
bigges protein structure: "ribosome": 109.02
'''
CZII_SMALLEST_PROTEIN_SIZE = 33.27
