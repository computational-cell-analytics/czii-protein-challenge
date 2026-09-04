#! /bin/bash
#SBATCH -c 16
#SBATCH --mem 128G
#SBATCH -p grete:shared
#SBATCH -t 12:00:00
#SBATCH -G A100:1
#SBATCH --output=/user/muth9/u12095/czii-protein-challenge/classification/slurm/slurm-%j.out
#SBATCH -A nim00007

# Arm 2: same contrastive recipe, but the encoder sees the REAL tomograms (unlabeled).
# --max-extent 33 is required: this dataset has no max_extent.json, and the default 39 would
# give 43-voxel crops instead of the 37 the classifier is fine-tuned and run on.
/mnt/vast-nhr/home/muth9/u12095/micromamba/envs/pro-revelio/bin/python /user/muth9/u12095/czii-protein-challenge/classification_scripts/pretrain_contrastive.py \
    --name protein_classification_contrastive_real_v1 \
    --datasets tomograms_VoxelSpacing10 \
    --max-extent 33 \
    --iterations 10000 \
    --batch-size 64 \
    --num-workers 16
