#!/bin/bash
# STA average at the validated GROUND-TRUTH orientations for ONE species.
# Reconstruct only (no angular search) -> fast, CPU-only.
#   bash run_gt_average.sh <basic|faket> <species>
# Outputs are written into the dataset dir (NOT here):
#   <dataset>/relion_sta[_faket]/<species>/GTaverage/run_it001_class001.mrc
set -e
VARIANT=$1; SP=$2
[ -z "$SP" ] && { echo "usage: bash run_gt_average.sh <basic|faket> <species>"; exit 1; }
DATASET=/mnt/vast-nhr/projects/nim00007/simulated_cryoET/czii_dataset_20260423
case "$VARIANT" in
  basic) PROJ=$DATASET/relion_sta ;;
  faket) PROJ=$DATASET/relion_sta_faket ;;
  *) echo "variant must be 'basic' or 'faket'"; exit 1 ;;
esac
source "$(cd "$(dirname "$0")" && pwd)/env.sh"
declare -A DIAM=( [vlp]=320 [ribosome]=300 [beta-galactosidase]=180 [thyroglobulin]=250 \
                  [apo-ferritin]=130 [beta-amylase]=130 [albumin]=100 )
D=${DIAM[$SP]:-300}
cd "$PROJ/$SP"
mkdir -p GTaverage
relion_refine --i particles.star --o GTaverage/run \
  --ref initial_ref.mrc --ini_high 40 --skip_align --iter 1 \
  --sym C1 --particle_diameter "$D" --flatten_solvent --zero_mask \
  --pad 2 --tau2_fudge 4 --dont_combine_weights_via_disc --pool 8 --j 8
echo "DONE  ->  $PROJ/$SP/GTaverage/run_it001_class001.mrc"
