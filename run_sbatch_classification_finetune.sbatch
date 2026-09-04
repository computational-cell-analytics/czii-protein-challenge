#! /bin/bash
#SBATCH -c 4
#SBATCH --mem 256G
#SBATCH -p grete:shared
#SBATCH -t 3:00:00
#SBATCH -G A100:1
#SBATCH --output=/user/muth9/u12095/czii-protein-challenge/classification/slurm/slurm-%j.out
#SBATCH -A nim00007

# Stage 2: supervised fine-tuning from the contrastively pretrained encoder.
# --split-from keeps the train/val/test tomograms identical to the v71 baseline.
PY=/mnt/vast-nhr/home/muth9/u12095/micromamba/envs/pro-revelio/bin/python
ENCODER=/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/models/checkpoints/protein_classification_contrastive_v1
SPLIT=/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/training/protein_classification_czii_v71

$PY /user/muth9/u12095/czii-protein-challenge/classification_scripts/train_classification.py \
    --loss focal \
    --name protein_classification_czii_v72_contrastive \
    --split-from $SPLIT \
    --pretrained-encoder $ENCODER \
    --pretrained-use-teacher \
    --encoder-lr-scale 0.1
