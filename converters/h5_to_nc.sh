#!/usr/bin/env bash
set -euo pipefail

IN="${1:?Usage: $0 input.h5 output.nc [logfile]}"
OUT="${2:?Usage: $0 input.h5 output.nc [logfile]}"
LOG="${3:-h5_to_nc.log}"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

TMP="${OUT}.tmp.$$"

echo "[INFO] $(date -Is) starting: $IN -> $OUT" | tee -a "$LOG"

python "$SCRIPT_DIR/h5_to_nc.py" \
  --in_h5 "$IN" \
  --out_nc "$TMP" \
  "${@:4}" 2>&1 | tee -a "$LOG"

mv -f "$TMP" "$OUT"
touch "${OUT}.done"
echo "[INFO] $(date -Is) done: $OUT" | tee -a "$LOG"

