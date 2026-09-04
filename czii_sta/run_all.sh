#!/bin/bash
# Submit the whole CZII ribosome STA pipeline as a SLURM dependency chain.
# Each job starts only if the previous one succeeded (afterok), so a failure
# anywhere stops the rest instead of running on bad inputs.
#
#   bash run_all.sh            # all five steps
#   bash run_all.sh 3          # start from step 3 (no dependency on 1-2)
#   MASK_THRESHOLD=0.012 bash run_all.sh   # override the automatic threshold
#
# Nothing is executed here -- these are submissions. Watch with:
#   squeue -u $USER   /   tail -f slurm_logs/slurm-<jobid>.out
set -e
cd "$(dirname "$0")"
START=${1:-1}

STEPS=(run_sbatch_extract.sbatch
       run_sbatch_initialmodel.sbatch
       run_sbatch_refine3d.sbatch
       run_sbatch_mask.sbatch
       run_sbatch_postprocess.sbatch)
NAMES=(extract initialmodel refine3d mask postprocess)

# MASK_THRESHOLD must reach the mask job's environment, not just this shell.
EXPORT=""
[ -n "$MASK_THRESHOLD" ] && EXPORT="--export=ALL,MASK_THRESHOLD=$MASK_THRESHOLD"

dep=""
for i in $(seq $((START-1)) 4); do
    # --kill-on-invalid-dep: if an upstream step fails, SLURM removes the rest
    # instead of leaving them queued forever as DependencyNeverSatisfied.
    out=$(sbatch $dep --kill-on-invalid-dep=yes $EXPORT "${STEPS[$i]}")
    jid=$(echo "$out" | awk '{print $NF}')
    printf "  %-13s %-34s job %s%s\n" "${NAMES[$i]}" "${STEPS[$i]}" "$jid" \
           "$([ -n "$dep" ] && echo "  (after $prev)")"
    dep="--dependency=afterok:$jid"
    prev=$jid
done
echo
echo "submitted. squeue -u $USER"
