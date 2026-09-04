#! /bin/bash
#SBATCH -c 8
#SBATCH --mem 64G
#SBATCH -p grete:shared
#SBATCH -t 3:00:00
#SBATCH -G A100:1
#SBATCH --output=/user/muth9/u12095/czii-protein-challenge/classification/slurm/slurm-%j.out
#SBATCH -A nim00007

# Stage 3: noise-robustness sweep of the baseline against the contrastively pretrained model,
# on the identical test tomograms.
PY=/mnt/vast-nhr/home/muth9/u12095/micromamba/envs/pro-revelio/bin/python
CKPT=/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/models/checkpoints
SPLIT=/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/training/protein_classification_czii_v71

$PY /user/muth9/u12095/czii-protein-challenge/classification_scripts/evaluation/eval_noise_robustness.py \
    --checkpoints $CKPT/protein_classification_czii_v71 \
                  $CKPT/protein_classification_czii_v72_contrastive \
    --split-dir $SPLIT \
    --datasets ExperimentRuns_faket_dens1_5_distr_eqCl3 \
    --num-workers 8 \
    -o /user/muth9/u12095/czii-protein-challenge/classification_scripts/evaluation/results/noise_robustness_v71_vs_v72.csv
