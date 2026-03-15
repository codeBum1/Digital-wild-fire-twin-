#!/bin/bash
# Polls build job 81644; submits fine-tune + chained eval once it completes.
BUILD_JOB=81644
INTERVAL=60

echo "Watching job $BUILD_JOB..."
while squeue -j $BUILD_JOB -h 2>/dev/null | grep -q "$BUILD_JOB"; do
    sleep $INTERVAL
done

echo "Build job $BUILD_JOB finished at $(date)"
echo "Final shard count: $(ls /home/abazan/wrfout_sandbox/vit_dataset_real_fire_tplus1/shards/ | wc -l)"
tail -10 /home/abazan/wrfout_sandbox/build_real_fire.out

# Submit fine-tune; chain eval after it completes
TRAIN_JOB=$(sbatch /home/abazan/wrfout_sandbox/train_real_fire.slurm | awk '{print $NF}')
echo "Submitted fine-tune job: $TRAIN_JOB"

EVAL_JOB=$(sbatch --dependency=afterok:$TRAIN_JOB /home/abazan/wrfout_sandbox/eval_evans_retrained.slurm | awk '{print $NF}')
echo "Submitted eval job: $EVAL_JOB (runs after $TRAIN_JOB)"

echo "Monitor with: squeue -u abazan"
