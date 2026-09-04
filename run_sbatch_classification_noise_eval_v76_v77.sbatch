#! /bin/bash
#SBATCH -c 8
#SBATCH --mem 64G
#SBATCH -p grete:shared
#SBATCH -t 3:00:00
#SBATCH -G A100:1
#SBATCH --output=/user/muth9/u12095/czii-protein-challenge/classification/slurm/slurm-%j.out
#SBATCH -A nim00007

/mnt/vast-nhr/home/muth9/u12095/micromamba/envs/pro-revelio/bin/python /user/muth9/u12095/czii-protein-challenge/classification_scripts/evaluation/eval_noise_robustness.py \
    --checkpoints /mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/models/checkpoints/protein_classification_czii_v71 \
                  /mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/models/checkpoints/protein_classification_czii_v75_contrastive \
                  /mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/models/checkpoints/protein_classification_czii_v76_contrastive_nrcl \
                  /mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/models/checkpoints/protein_classification_czii_v77_contrastive_supcon \
    --split-dir /mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/training/protein_classification_czii_v71 \
    --datasets ExperimentRuns_faket_dens1_5_distr_eqCl3 \
    --num-workers 8 \
    -o /user/muth9/u12095/czii-protein-challenge/classification_scripts/evaluation/results/noise_robustness_v71_v75_v76_v77.csv
