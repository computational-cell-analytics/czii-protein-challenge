#! /bin/bash
#SBATCH -c 4
#SBATCH --mem 32G
#SBATCH -p standard96s:shared
#SBATCH -t 0:30:00
#SBATCH --output=/user/muth9/u12095/czii-protein-challenge/czii_sta/slurm_logs/slurm-%j.out
#SBATCH -A nim00007

source /user/muth9/u12095/czii-protein-challenge/czii_sta/env_cpu.sh
command -v relion_postprocess >/dev/null || { echo "ERROR: relion not on PATH"; exit 1; }

cd /mnt/vast-nhr/projects/nim00007/simulated_cryoET/czii_sta_ribosome
mkdir -p PostProcess

# The corrected FSC in postprocess.star / postprocess_fsc.eps is the only
# defensible resolution number: it subtracts, via phase randomisation, whatever
# correlation the mask itself creates.
relion_postprocess \
    --i Refine3D/run_half1_class001_unfil.mrc \
    --mask Mask/mask.mrc \
    --o PostProcess/postprocess \
    --auto_bfac
