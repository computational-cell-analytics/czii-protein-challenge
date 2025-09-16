#! /bin/bash
#SBATCH -c 4 #4 #8
#SBATCH --mem 120G #120G #32G #64G #256G
#SBATCH -p grete-h100:shared
#SBATCH -t 32:00:00 #6:00:00 #48:00:00 
#SBATCH -G H100:1 #V100:1 #2 #A100:1  #gtx1080:2 #v100:1 #H100:1
#SBATCH --output=/user/muth9/u12095/czii-protein-challenge/classification/slurm/slurm-%j.out 
#SBATCH -A nim00007 #SBATCH --constraint 80gb 


/scratch/usr/nimsmuth/mambaforge/envs/synapse-net/bin/python /user/muth9/u12095/czii-protein-challenge/classification/training/train_classification.py