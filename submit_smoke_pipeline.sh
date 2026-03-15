#!/bin/bash
# Submit the full smoke pipeline:
#   1. build_smoke_fire  (real-fire corpus, ~8h)
#   2. build_evans_smoke (Evans Canyon, ~2h)  -- runs in parallel with #1
#   3. train_smoke       (after both builds finish)
#   4. eval_evans_smoke  (after training finishes)

set -e
cd ~/wrfout_sandbox

BUILD_FIRE=$(sbatch build_smoke_fire.slurm | awk '{print $NF}')
echo "Submitted build_smoke_fire:  job $BUILD_FIRE"

BUILD_EVANS=$(sbatch build_evans_smoke.slurm | awk '{print $NF}')
echo "Submitted build_evans_smoke: job $BUILD_EVANS"

TRAIN_JOB=$(sbatch --dependency=afterok:${BUILD_FIRE}:${BUILD_EVANS} train_smoke.slurm | awk '{print $NF}')
echo "Submitted train_smoke:       job $TRAIN_JOB  (after $BUILD_FIRE + $BUILD_EVANS)"

EVAL_JOB=$(sbatch --dependency=afterok:${TRAIN_JOB} eval_evans_smoke.slurm | awk '{print $NF}')
echo "Submitted eval_evans_smoke:  job $EVAL_JOB  (after $TRAIN_JOB)"

echo ""
echo "Pipeline queued. Monitor with: squeue -u $USER"
echo "Final results will appear in: eval_evans_smoke.out"
