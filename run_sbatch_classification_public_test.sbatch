#! /bin/bash
#SBATCH -c 8
#SBATCH --mem 256G
#SBATCH -p grete:shared
#SBATCH -t 12:00:00
#SBATCH -G A100:1
#SBATCH --output=/user/muth9/u12095/czii-protein-challenge/classification/slurm/slurm-%j.out
#SBATCH -A nim00007

# Classify the real public test tomograms with model $1 and write the confusion matrices /
# classification_report_ALL.txt. Usage: sbatch <this> <model_name>
MODEL=$1
R=/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge
OUT=$R/classification/$MODEL/public_test_dataset

/mnt/vast-nhr/home/muth9/u12095/micromamba/envs/pro-revelio/bin/python /user/muth9/u12095/czii-protein-challenge/classification_scripts/run_protein_classification.py \
    -i $R/data/tomograms_VoxelSpacing10 \
    -mlp \
    -o $OUT \
    -m $R/models/checkpoints/$MODEL/ \
    -sub_o $OUT/subtomograms \
    -l $R/ground_truth/structure_for_detection
