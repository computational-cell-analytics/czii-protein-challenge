#! /bin/bash
#SBATCH -c 4
#SBATCH --mem 256G
#SBATCH -p grete:shared
#SBATCH -t 3:00:00
#SBATCH -G A100:1
#SBATCH --output=/user/muth9/u12095/czii-protein-challenge/detection/slurm/slurm-%j.out
#SBATCH -A nim00007

# Unsupervised domain adaptation of the v36 detection model to the real
# ExperimentRuns tomograms (no labels used; teacher/student consistency only).
# To run semi-supervised instead, add:
#   --label_root /mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/ground_truth/structure_for_detection/ExperimentRuns
/mnt/vast-nhr/home/muth9/u12095/micromamba/envs/synapse-net/bin/python \
 /user/muth9/u12095/czii-protein-challenge/detection_scripts/train_detection_domain_adaptation.py \
 -n protein_detection_da_v34_ExperimentRuns \
 -i /mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/data/ExperimentRuns \
 -m /mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/models/checkpoints/protein_detection_czii_v34
