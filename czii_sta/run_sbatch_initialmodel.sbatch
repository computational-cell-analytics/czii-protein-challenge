#! /bin/bash
#SBATCH -c 8
#SBATCH --mem 64G
#SBATCH -p grete:shared
#SBATCH -t 8:00:00
#SBATCH -G A100:1
#SBATCH --output=/user/muth9/u12095/czii-protein-challenge/czii_sta/slurm_logs/slurm-%j.out
#SBATCH -A nim00007

source /user/muth9/u12095/czii-protein-challenge/czii_sta/env_gpu.sh
command -v relion_refine >/dev/null || { echo "ERROR: relion not on PATH"; exit 1; }

cd /mnt/vast-nhr/projects/nim00007/simulated_cryoET/czii_sta_ribosome
mkdir -p InitialModel

# The picks carry no orientations, so the starting model must come from the data.
# C1 throughout: a ribosome is asymmetric, nothing to impose or align afterwards.
relion_refine --grad --denovo_3dref \
    --i particles.star \
    --o InitialModel/run \
    --ctf \
    --iter 100 \
    --K 1 --sym C1 \
    --particle_diameter 300 \
    --flatten_solvent --zero_mask \
    --healpix_order 2 \
    --offset_range 6 --offset_step 2 \
    --pool 16 \
    --j 8 --gpu 0
