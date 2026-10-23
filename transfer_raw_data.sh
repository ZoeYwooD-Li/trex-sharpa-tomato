#!/usr/bin/env bash
# Copy only the four training datasets, following the two source symlinks.
set -euo pipefail

: "${SOURCE_ROOT:?Set the directory containing A/ and B/}"
: "${DEST_ROOT:?Set an existing local directory or rsync SSH destination}"

cd "$SOURCE_ROOT"
for dataset in \
  A/121_430_data_lerobot \
  A/54_120_data_lerobot \
  B/130_423_data_lerobot \
  B/56_106_data_lerobot; do
  test -f "$dataset/meta/info.json" || { echo "Missing $dataset/meta/info.json" >&2; exit 1; }
done

rsync -aL --partial --info=progress2 --relative \
  A/121_430_data_lerobot \
  A/54_120_data_lerobot \
  B/130_423_data_lerobot \
  B/56_106_data_lerobot \
  "$DEST_ROOT"
