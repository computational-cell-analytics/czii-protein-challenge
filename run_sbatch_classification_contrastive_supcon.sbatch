#! /bin/bash
#SBATCH -c 16
#SBATCH --mem 128G
#SBATCH -p grete:shared
#SBATCH -t 12:00:00
#SBATCH -G A100:1
#SBATCH --output=/user/muth9/u12095/czii-protein-challenge/classification/slurm/slurm-%j.out
#SBATCH -A nim00007

# Negative-selection variant "supervised". Identical to v75's pretraining in every other respect:
# same eqCl3 dataset, same split, same 10000 iterations, teacher-eval fix applied.
/mnt/vast-nhr/home/muth9/u12095/micromamba/envs/pro-revelio/bin/python /user/muth9/u12095/czii-protein-challenge/classification_scripts/pretrain_contrastive.py \
    --name protein_classification_contrastive_supcon_v1 \
    --negatives supervised  \
    --split-from /mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/training/protein_classification_czii_v71 \
    --iterations 10000 \
    --batch-size 64 \
    --num-workers 16
