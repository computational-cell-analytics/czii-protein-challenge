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
  experimental) PROJ=/mnt/vast-nhr/projects/nim00007/simulated_cryoET/czii_dataset_20260423/relion_sta_experimental ;;
  *) echo "variant must be 'basic', 'faket', or 'experimental'"; exit 1 ;;
esac
ROOT="$(cd "$(dirname "$0")" && pwd)"
source "$ROOT/env.sh"
declare -A DIAM=( [vlp]=320 [virus-like-particle]=320 [ribosome]=300 [beta-galactosidase]=180 \
                  [thyroglobulin]=250 [apo-ferritin]=130 [beta-amylase]=130 [albumin]=100 )
D=${DIAM[$SP]:-300}
cd "$PROJ/$SP"

# --- restart handling ---------------------------------------------------------
# already finished? skip.
if [ -f Refine3D/run_class001.mrc ]; then
  echo "SKIP $SP: already finished (Refine3D/run_class001.mrc exists). Delete Refine3D/ to redo."
  exit 0
fi
mkdir -p Refine3D
# interrupted? continue from the latest COMPLETED iteration (it001+).
# NB: never continue from it000 -- that is RELION's pre-refinement state and its
# stored reference name is empty, so --continue from it fails ("empty file name").
LAST=$(ls -1 Refine3D/run_it*_optimiser.star 2>/dev/null | grep -v '_it000_optimiser' | sort -V | tail -1)
if [ -n "$LAST" ]; then
  echo "CONTINUE $SP from $LAST"
  mpirun --oversubscribe -n 3 relion_refine_mpi --continue "$LAST" --o Refine3D/run \
    --dont_combine_weights_via_disc --pool 8 --j 6 --gpu ""
else
  echo "START $SP (fresh)"
  rm -f Refine3D/run_it000_* Refine3D/run_ct* 2>/dev/null   # clear any incomplete it000 from a cancelled start
  rm -f Refine3D/run_half*_class*_unfil.mrc 2>/dev/null
  # -n 3 = 1 leader + 2 half-set followers (gold-standard).
  # --oversubscribe: the interactive SLURM allocation exposes <3 MPI "slots".
  mpirun --oversubscribe -n 3 relion_refine_mpi --i particles.star --o Refine3D/run \
    --ref initial_ref.mrc --ini_high 50 --firstiter_cc \
    --auto_refine --split_random_halves --sym C1 \
    --particle_diameter "$D" --flatten_solvent --zero_mask \
    --oversampling 1 --healpix_order 2 --auto_local_healpix_order 4 \
    --offset_range 8 --offset_step 2 --pad 2 \
    --dont_combine_weights_via_disc --pool 8 --j 6 --gpu ""
fi
echo "DONE  ->  $PROJ/$SP/Refine3D/run_class001.mrc   (resolution in run_model.star)"
