#!/usr/bin/env bash
# Full 1080p render in restartable parts (finished parts are kept), then a lossless concat.
set -euo pipefail
cd "$(dirname "$0")"; mkdir -p out
BOUNDS=(0 17 31 47 64 80 98.5 110.5 131.5 146)
for ((i = 0; i < ${#BOUNDS[@]} - 1; i++)); do
  f="out/part_$i.mp4"
  [ -s "$f" ] || { node render.mjs --start "${BOUNDS[$i]}" --end "${BOUNDS[$((i + 1))]}" --out "out/part_$i.tmp.mp4" && mv "out/part_$i.tmp.mp4" "$f"; }
done
ls out/part_?.mp4 | sed "s|^|file '$PWD/|; s|$|'|" > out/parts.txt
ffmpeg -y -loglevel error -f concat -safe 0 -i out/parts.txt -c copy -movflags +faststart ../results_v9/animation/ris_uav_3d.mp4
echo "ALLDONE $(date -u +%H:%M:%S)"
