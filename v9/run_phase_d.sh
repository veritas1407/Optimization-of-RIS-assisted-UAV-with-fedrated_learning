#!/usr/bin/env bash
# Phase D training: learned policy without any RIS, and with the aerial (UAV-carried) RIS.
set -euo pipefail
SEEDS=${SEEDS-"0 1 2"}
N_PAR=${N_PAR:-4}
LOGDIR=results_v9/logs/phase_d
mkdir -p "$LOGDIR"
jobs=()
for s in $SEEDS; do
  jobs+=("building_noris $s")
  jobs+=("aerial $s")
done
[ ${#jobs[@]} -eq 0 ] && { echo "no jobs"; exit 0; }
printf '%s\n' "${jobs[@]}" | xargs -P "$N_PAR" -L1 bash -c \
  'python3 -m v9.run --exp train --skip-existing --episodes 3000 --algo ctde --preset "$0" --seed "$1" \
     > '"$LOGDIR"'/"$0"_s"$1".log 2>&1; grep -E "TEST|skipping|Error" '"$LOGDIR"'/"$0"_s"$1".log'
python3 -m v9.plots train building_noris
python3 -m v9.plots train aerial
