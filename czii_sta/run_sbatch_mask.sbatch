#! /bin/bash
#SBATCH -c 4
#SBATCH --mem 32G
#SBATCH -p standard96s:shared
#SBATCH -t 0:30:00
#SBATCH --output=/user/muth9/u12095/czii-protein-challenge/czii_sta/slurm_logs/slurm-%j.out
#SBATCH -A nim00007
set -e

# CZII ships no masks, so this one is built from the refined map. The threshold is
# picked by Otsu on the map histogram and sanity-checked against the expected
# ribosome diameter; override by exporting MASK_THRESHOLD before submitting.
EXTEND=3             # grow the binary mask by 3 px (30 A)
SOFT_EDGE=6          # raised-cosine falloff over 6 px (60 A)

source /user/muth9/u12095/czii-protein-challenge/czii_sta/env_cpu.sh
command -v relion_mask_create >/dev/null || { echo "ERROR: relion not on PATH"; exit 1; }

cd /mnt/vast-nhr/projects/nim00007/simulated_cryoET/czii_sta_ribosome
mkdir -p Mask

if [ -n "$MASK_THRESHOLD" ]; then
  THRESHOLD=$MASK_THRESHOLD
  echo "using MASK_THRESHOLD from the environment: $THRESHOLD"
else
  THRESHOLD=$(/mnt/vast-nhr/home/muth9/u12095/micromamba/envs/pro-revelio/bin/python /user/muth9/u12095/czii-protein-challenge/czii_sta/pick_mask_threshold.py \
                  --i Refine3D/run_class001.mrc --expect-diameter 300)
fi
echo "threshold = $THRESHOLD" | tee Mask/threshold.txt

relion_mask_create \
    --i Refine3D/run_class001.mrc \
    --o Mask/mask.mrc \
    --ini_threshold "$THRESHOLD" \
    --extend_inimask $EXTEND \
    --width_soft_edge $SOFT_EDGE \
    --j 4
