#! /bin/bash
#SBATCH -c 8
#SBATCH --mem 64G
#SBATCH -p standard96s:shared
#SBATCH -t 4:00:00
#SBATCH --output=/user/muth9/u12095/czii-protein-challenge/czii_sta/slurm_logs/slurm-%j.out
#SBATCH -A nim00007

# Pure python (zarr + mrcfile), no RELION and no GPU -> CPU partition.
/mnt/vast-nhr/home/muth9/u12095/micromamba/envs/pro-revelio/bin/python /user/muth9/u12095/czii-protein-challenge/czii_sta/extract_particles.py \
    --species ribosome \
    --box 64 \
    --fraction 0.15 \
    --seed 0 \
    --out /mnt/vast-nhr/projects/nim00007/simulated_cryoET/czii_sta_ribosome
