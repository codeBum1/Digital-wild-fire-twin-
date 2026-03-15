#!/bin/bash
# Collect symlinks to all usable PNW WRF-SFIRE runs (≥96 wrfout = ≥48h of 72h sim).
# Creates pnw_sfire_usable/ where each entry is a symlink to a run's wrf/ dir,
# matching the layout build_pnw_dataset.py expects (PNW_RAW_DIR with wrfout_d01_* inside).

set -e

OUT_DIR=/home/abazan/wrfout_sandbox/pnw_sfire_usable
mkdir -p "$OUT_DIR"

BATCHES="20260301_50km 20260302_set3 20260303_ace 20260303_ace2 20260303_set5"
DOMAIN="d01"
MIN_WRFOUT=96   # ≥48 h of the 72 h simulation

n_full=0; n_partial=0; n_skipped=0

for batch in $BATCHES; do
    base="/scratch/wdt/pnw_sfire_dataset/runs/$batch"
    for run_dir in "$base"/*/; do
        wrf_dir="$run_dir/wrf"
        n=$(ls "$wrf_dir/wrfout_${DOMAIN}_"* 2>/dev/null | wc -l)
        sid=$(basename "$run_dir")

        if [ "$n" -ge 145 ]; then
            ln -sfn "$wrf_dir" "$OUT_DIR/${sid}"
            n_full=$((n_full + 1))
        elif [ "$n" -ge $MIN_WRFOUT ]; then
            ln -sfn "$wrf_dir" "$OUT_DIR/${sid}"
            n_partial=$((n_partial + 1))
        else
            n_skipped=$((n_skipped + 1))
        fi
    done
done

total=$((n_full + n_partial))
echo "Symlinks created: $total  (full=$n_full  partial=$n_partial  skipped=$n_skipped)"
echo "Output dir: $OUT_DIR"
ls "$OUT_DIR" | wc -l
