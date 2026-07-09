#! /bin/bash
#SBATCH -c 4
#SBATCH --mem 256G
#SBATCH -p grete:shared
#SBATCH -t 3:00:00
#SBATCH -G A100:1
#SBATCH --output=/user/muth9/u12095/czii-protein-challenge/classification/slurm/slurm-%j.out
#SBATCH -A nim00007

# Unsupervised domain adaptation of the v62 classification model to the real
# ExperimentRuns tomograms. Candidate crop coordinates are taken from the point
# annotations in ground_truth/structure_for_detection (their class labels are
# ignored; the classes are pseudo-labeled by the teacher).
/mnt/vast-nhr/home/muth9/u12095/micromamba/envs/pro-revelio/bin/python \
 /user/muth9/u12095/czii-protein-challenge/classification_scripts/train_classification_domain_adaptation.py \
 -n protein_classification_da_v62_ExperimentRuns \
 -m /mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/models/checkpoints/protein_classification_czii_v62 \
 --source_idx_to_label /mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/training/protein_classification_czii_v62/idx_to_label.json \
 --datasets ExperimentRuns \
 --coords_source annotation_json \
 --target_root /mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/ground_truth/structure_for_detection/
