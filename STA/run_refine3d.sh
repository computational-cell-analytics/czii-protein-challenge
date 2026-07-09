#!/bin/bash
# Full de-novo 3D auto-refine for ONE species (RELION searches orientations itself;
# GT angles in the star are ignored). GPU + MPI. Run on the Jupyter desktop GPU node.
#   source env.sh                          # once per shell
#   bash run_refine3d.sh <basic|faket> <species>
# Outputs: <dataset>/relion_sta[_faket]/<species>/Refine3D/run_class001.mrc
set -e
VARIANT=$1; SP=$2
[ -z "$SP" ] && { echo "usage: bash run_refine3d.sh <basic|faket> <species>"; exit 1; }
DATASET=/mnt/vast-nhr/projects/nim00007/simulated_cryoET/czii_dataset_20260423
case "$VARIANT" in
  basic) PROJ=$DATASET/relion_sta ;;
  faket) PROJ=$DATASET/relion_sta_faket ;;
  *) echo "variant must be 'basic' or 'faket'"; exit 1 ;;
esac
ROOT="$(cd "$(dirname "$0")" && pwd)"
source "$ROOT/env.sh"
declare -A DIAM=( [vlp]=320 [ribosome]=300 [beta-galactosidase]=180 [thyroglobulin]=250 \
                  [apo-ferritin]=130 [beta-amylase]=130 [albumin]=100 )
D=${DIAM[$SP]:-300}
cd "$PROJ/$SP"
mkdir -p Refine3D
# -n 3 = 1 leader + 2 half-set followers (gold-standard). 3 RTX5000 available on agq004.
mpirun -n 3 relion_refine_mpi --i particles.star --o Refine3D/run \
  --ref initial_ref.mrc --ini_high 50 \
  --auto_refine --split_random_halves --sym C1 \
  --particle_diameter "$D" --flatten_solvent --zero_mask \
  --oversampling 1 --healpix_order 2 --auto_local_healpix_order 4 \
  --offset_range 8 --offset_step 2 --pad 2 --tau2_fudge 4 \
  --dont_combine_weights_via_disc --pool 8 --j 6 --gpu ""
echo "DONE  ->  $PROJ/$SP/Refine3D/run_class001.mrc   (resolution in run_model.star)"
