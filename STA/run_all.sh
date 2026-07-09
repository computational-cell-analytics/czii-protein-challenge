#!/bin/bash
# Run all 7 species one after another, for one variant.
#   bash run_all.sh <gt|refine> <basic|faket>
#   bash run_all.sh gt basic       # GT-angle averages, plain-noise tomos
#   bash run_all.sh gt faket       # GT-angle averages, style-transfer tomos
#   bash run_all.sh refine basic   # de-novo auto-refine (GPU+MPI)
MODE=${1:-gt}; VARIANT=${2:-basic}
ROOT="$(cd "$(dirname "$0")" && pwd)"
for SP in ribosome vlp beta-galactosidase thyroglobulin apo-ferritin beta-amylase albumin; do
  echo "======== $SP  ($MODE, $VARIANT) ========"
  if [ "$MODE" = gt ]; then bash "$ROOT/run_gt_average.sh" "$VARIANT" "$SP"
  else bash "$ROOT/run_refine3d.sh" "$VARIANT" "$SP"; fi
done
echo "ALL DONE"
