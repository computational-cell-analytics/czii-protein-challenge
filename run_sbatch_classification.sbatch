#! /bin/bash
#SBATCH -c 4 #4 #8
#SBATCH --mem 32G #120G #32G #64G #256G
#SBATCH -p grete:shared
#SBATCH -t 1:00:00 #6:00:00 #48:00:00 
#SBATCH -G A100:1 #V100:1 #2 #A100:1  #gtx1080:2 #v100:1 #H100:1
#SBATCH --output=/user/muth9/u12095/czii-protein-challenge/classification/slurm/slurm-%j.out 
#SBATCH -A nim00007 #SBATCH --constraint 80gb 


/scratch/usr/nimsmuth/mambaforge/envs/synapse-net/bin/python /user/muth9/u12095/czii-protein-challenge/classification/inference/run_protein_classification.py \
 -i /scratch-grete/projects/nim00007/cryo-et/challenge-data/train/static/ExperimentRuns/TS_73_6 \
 -o /mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/classification/protein_classification_czii_v4 \
 -m /mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/models/checkpoints/protein_classification_czii_v4/ \
 --max_extent 35 \
 -sub_o /mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/classification/protein_classification_czii_v4/subtomograms \
 -l /scratch-grete/projects/nim00007/cryo-et/challenge-data/train/overlay/ExperimentRuns/