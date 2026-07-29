#!/bin/bash
# Submit ONE GPU batch job per species for de-novo 3D auto-refine, so runs don't
# depend on the interactive desktop's time limit. Each job calls run_refine3d.sh,
# which SKIPS finished species and RESUMES interrupted ones from their checkpoint
# -> safe to resubmit after a job hits its walltime.
#
#   bash sbatch_refine.sh experimental                       # all species
#   bash sbatch_refine.sh experimental ribosome apo-ferritin # just these
#   bash sbatch_refine.sh basic                              # sim variant
#
# Adjust -p / -A / -G / --time for your cluster if needed.
VARIANT=${1:?usage: bash sbatch_refine.sh <basic|faket|experimental> [species...]}; shift
ROOT="$(cd "$(dirname "$0")" && pwd)"
if [ "$#" -gt 0 ]; then
  SPECIES="$*"
elif [ "$VARIANT" = experimental ]; then
  SPECIES="ribosome virus-like-particle beta-galactosidase thyroglobulin apo-ferritin beta-amylase"
else
  SPECIES="ribosome vlp beta-galactosidase thyroglobulin apo-ferritin beta-amylase albumin"
fi
mkdir -p "$ROOT/slurm_logs"
for SP in $SPECIES; do
  jid=$(sbatch --parsable \
    --job-name="ref_${VARIANT}_${SP}" \
    -p grete:shared -G A100:2 --nodes=1 --ntasks=3 --cpus-per-task=6 --mem=96G \
    --time=24:00:00 \
    -o "$ROOT/slurm_logs/refine_${VARIANT}_${SP}_%j.out" \
    --wrap="bash '$ROOT/run_refine3d.sh' '$VARIANT' '$SP'")
  echo "submitted $VARIANT/$SP as job $jid"
done
echo "watch: squeue --me ; logs in $ROOT/slurm_logs/"
