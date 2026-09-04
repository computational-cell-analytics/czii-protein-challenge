#! /bin/bash
#SBATCH -c 4
#SBATCH --mem 256G
#SBATCH -p grete:shared
#SBATCH -t 3:00:00
#SBATCH -G A100:1
#SBATCH --output=/user/muth9/u12095/czii-protein-challenge/classification/slurm/slurm-%j.out
#SBATCH -A nim00007

# Arm 2: encoder pretrained on the REAL tomograms, fine-tuned on the simulated labels.
/mnt/vast-nhr/home/muth9/u12095/micromamba/envs/pro-revelio/bin/python /user/muth9/u12095/czii-protein-challenge/classification_scripts/train_classification.py \
    --loss focal \
    --name protein_classification_czii_v74_contrastive_real \
    --split-from /mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/training/protein_classification_czii_v71 \
    --pretrained-encoder /mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/models/checkpoints/protein_classification_contrastive_real_v1 \
    --pretrained-use-teacher \
    --encoder-lr-scale 1.0
