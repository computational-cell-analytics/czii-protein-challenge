#! /bin/bash
#SBATCH -c 8
#SBATCH --mem 256G
#SBATCH -p grete:shared
#SBATCH -t 2:00:00
#SBATCH -G A100:1
#SBATCH --output=/user/muth9/u12095/czii-protein-challenge/classification/slurm/slurm-%j.out
#SBATCH -A nim00007

# Controlled noise-model shift: same simulated scene, same particles, byte-identical pick
# coordinates as the training set -- only the noise/reconstruction model differs.
# --split_file restricts to v71's 5 test tomograms, so it is the same 1626 crops v71 scored on.
MODEL=$1
R=/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge
OUT=$R/classification/$MODEL/ExperimentRuns_basicNoise_dens1_5_distr_eqCl3

/mnt/vast-nhr/home/muth9/u12095/micromamba/envs/pro-revelio/bin/python /user/muth9/u12095/czii-protein-challenge/classification_scripts/run_protein_classification.py \
    -i $R/data/ExperimentRuns_basicNoise_dens1_5_distr_eqCl3 \
    -s $R/training/protein_classification_czii_v71/split-ExperimentRuns_faket_dens1_5_distr_eqCl3.json \
    -o $OUT \
    -m $R/models/checkpoints/$MODEL/ \
    -sub_o $OUT/subtomograms \
    -l $R/ground_truth/structure_for_detection
