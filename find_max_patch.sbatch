#! /bin/bash
#SBATCH -c 4
#SBATCH --mem 256G
#SBATCH -p grete:shared
#SBATCH -t 0:30:00
#SBATCH -G A100:1
#SBATCH --output=/user/muth9/u12095/czii-protein-challenge/detection/slurm/maxpatch-%j.out
#SBATCH -A nim00007

/mnt/vast-nhr/home/muth9/u12095/micromamba/envs/pro-revelio/bin/python /user/muth9/u12095/czii-protein-challenge/detection_scripts/find_max_patch.py
