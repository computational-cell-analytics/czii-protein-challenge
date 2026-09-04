#! /bin/bash
#SBATCH -c 8
#SBATCH --ntasks 3
#SBATCH --nodes 1
#SBATCH --mem 96G
#SBATCH -p grete:shared
#SBATCH -t 48:00:00
#SBATCH -G A100:2
#SBATCH --output=/user/muth9/u12095/czii-protein-challenge/czii_sta/slurm_logs/slurm-%j.out
#SBATCH -A nim00007

# ntasks 3 = leader + 2 half-set workers (the --split_random_halves minimum),
# one GPU each. Never raise ntasks without raising GPUs to match.
source /user/muth9/u12095/czii-protein-challenge/czii_sta/env_gpu.sh
command -v relion_refine_mpi >/dev/null || { echo "ERROR: relion not on PATH"; exit 1; }

cd /mnt/vast-nhr/projects/nim00007/simulated_cryoET/czii_sta_ribosome
mkdir -p Refine3D

# No --solvent_mask yet: the mask is built FROM this map afterwards.
mpirun -n "$SLURM_NTASKS" relion_refine_mpi \
    --i particles.star \
    --ref InitialModel/run_it100_class001.mrc \
    --o Refine3D/run \
    --auto_refine --split_random_halves \
    --ini_high 50 \
    --sym C1 \
    --ctf \
    --particle_diameter 300 \
    --flatten_solvent --zero_mask \
    --healpix_order 3 \
    --auto_local_healpix_order 5 \
    --auto_ignore_angles --auto_resol_angles \
    --offset_range 6 --offset_step 2 \
    --pool 16 \
    --dont_combine_weights_via_disc \
    --j 8 --gpu ""
