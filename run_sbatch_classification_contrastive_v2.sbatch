#! /bin/bash
#SBATCH -c 16
#SBATCH --mem 128G
#SBATCH -p grete:shared
#SBATCH -t 12:00:00
#SBATCH -G A100:1
#SBATCH --output=/user/muth9/u12095/czii-protein-challenge/classification/slurm/slurm-%j.out
#SBATCH -A nim00007

# Stage 1: contrastive pretraining. Crops are read lazily from zarr, hence the low --mem.
# Throughput is dataloader-bound (zarr reads + Fourier ops), so -c / --num-workers matter
# more here than the GPU.
PY=/mnt/vast-nhr/home/muth9/u12095/micromamba/envs/pro-revelio/bin/python
SPLIT=/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/training/protein_classification_czii_v71

$PY /user/muth9/u12095/czii-protein-challenge/classification_scripts/pretrain_contrastive.py \
    --name protein_classification_contrastive_v2 \
    --split-from $SPLIT \
    --iterations 10000 \
    --batch-size 64 \
    --num-workers 16

# Real "same particle, different SNR" positives; needs the paired SNR datasets, so this is
# NOT eqCl3-only:
# $PY .../pretrain_contrastive.py --name protein_classification_contrastive_crossSNR \
#     --datasets ExperimentRuns_faket_snr_0_12_0_2 --paired-datasets ExperimentRuns_basic_snr_0_1_0_15
