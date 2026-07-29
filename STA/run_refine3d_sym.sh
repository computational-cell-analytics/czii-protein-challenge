#!/bin/bash
# Symmetry-imposed de-novo 3D auto-refine for ONE species, written to a SEPARATE
# folder so it never clobbers the C1 run in Refine3D/. GPU + MPI.
#   bash run_refine3d_sym.sh <basic|faket|experimental> <species> <sym>
#   e.g. bash run_refine3d_sym.sh experimental virus-like-particle I
# Output: <dataset>/relion_sta[_...]/<species>/Refine3D_<sym>/run_class001.mrc
#
# Seeding: the icosahedral (or any point-group) refine must start from a reference
# whose symmetry axes sit on RELION's convention. We align the finished C1 map
# (Refine3D/run_class001.mrc) to the target symmetry with relion_align_symmetry;
# if no C1 map exists we fall back to aligning initial_ref.mrc.
set -e
VARIANT=$1; SP=$2; SYM=$3
[ -z "$SYM" ] && { echo "usage: bash run_refine3d_sym.sh <basic|faket|experimental> <species> <sym>"; exit 1; }
DATASET=/mnt/vast-nhr/projects/nim00007/simulated_cryoET/czii_dataset_20260423
case "$VARIANT" in
  basic) PROJ=$DATASET/relion_sta ;;
  faket) PROJ=$DATASET/relion_sta_faket ;;
  experimental) PROJ=$DATASET/relion_sta_experimental ;;
  *) echo "variant must be 'basic', 'faket', or 'experimental'"; exit 1 ;;
esac
ROOT="$(cd "$(dirname "$0")" && pwd)"
source "$ROOT/env.sh"
declare -A DIAM=( [vlp]=320 [virus-like-particle]=320 [ribosome]=300 [beta-galactosidase]=180 \
                  [thyroglobulin]=250 [apo-ferritin]=130 [beta-amylase]=130 [albumin]=100 )
D=${DIAM[$SP]:-300}
cd "$PROJ/$SP"
OUT="Refine3D_${SYM}"
mkdir -p "$OUT"

# already finished? skip.
if [ -f "$OUT/run_class001.mrc" ]; then
  echo "SKIP $SP ($SYM): already finished ($OUT/run_class001.mrc exists). Delete $OUT/ to redo."
  exit 0
fi

# --- build a symmetry-aligned starting reference (once) -----------------------
REF="$OUT/ref_${SYM}aligned.mrc"
if [ ! -f "$REF" ]; then
  if [ -f Refine3D/run_class001.mrc ]; then SRC=Refine3D/run_class001.mrc; else SRC=initial_ref.mrc; fi
  echo "ALIGN $SRC -> $REF  (--sym $SYM)"
  # NB: do NOT force --angpix; let it inherit the exact pixel size from the source
  # map header (matches the data). Forcing e.g. 10 when the data is 10.012 makes
  # relion_refine abort with a reference/data pixel-size mismatch.
  relion_align_symmetry --i "$SRC" --o "$REF" --sym "$SYM" --apply_sym
fi

# --- restart handling (continue from latest completed iter, never it000) ------
LAST=$(ls -1 "$OUT"/run_it*_optimiser.star 2>/dev/null | grep -v '_it000_optimiser' | sort -V | tail -1)
if [ -n "$LAST" ]; then
  echo "CONTINUE $SP ($SYM) from $LAST"
  mpirun --oversubscribe -n 3 relion_refine_mpi --continue "$LAST" --o "$OUT/run" \
    --dont_combine_weights_via_disc --pool 8 --j 6 --gpu ""
else
  echo "START $SP ($SYM) fresh"
  rm -f "$OUT"/run_it000_* "$OUT"/run_ct* 2>/dev/null
  rm -f "$OUT"/run_half*_class*_unfil.mrc 2>/dev/null
  mpirun --oversubscribe -n 3 relion_refine_mpi --i particles.star --o "$OUT/run" \
    --ref "$REF" --ini_high 50 --firstiter_cc \
    --auto_refine --split_random_halves --sym "$SYM" \
    --particle_diameter "$D" --flatten_solvent --zero_mask \
    --oversampling 1 --healpix_order 2 --auto_local_healpix_order 4 \
    --offset_range 8 --offset_step 2 --pad 2 \
    --dont_combine_weights_via_disc --pool 8 --j 6 --gpu ""
fi
echo "DONE  ->  $PROJ/$SP/$OUT/run_class001.mrc   (resolution in run_model.star)"
