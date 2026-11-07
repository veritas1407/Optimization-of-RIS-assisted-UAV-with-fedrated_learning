#!/usr/bin/env bash
# Runs the main training campaign (all methods x seeds), N_PAR runs in parallel.
#   bash v9/run_campaign.sh                 # building preset, seeds 0 1 2
#   PRESET=aerial SEEDS="0 1" bash v9/run_campaign.sh
set -euo pipefail
PRESET=${PRESET:-building}
SEEDS=${SEEDS:-"0 1 2"}
EPISODES=${EPISODES:-3000}
N_PAR=${N_PAR:-4}
LOGDIR=${LOGDIR:-results_v9/logs/$PRESET}
mkdir -p "$LOGDIR"

jobs=()
for s in $SEEDS; do
  jobs+=("--algo ctde --seed $s")
  jobs+=("--algo ippo --seed $s")
  jobs+=("--algo fedavg --E 1 --seed $s")
  jobs+=("--algo fedavg --E 4 --seed $s")
  jobs+=("--algo fedavg --E 16 --seed $s")
  jobs+=("--algo fedprox --E 4 --mu 0.01 --seed $s")
  jobs+=("--algo fedper --E 4 --seed $s")
done

printf '%s\n' "${jobs[@]}" | xargs -P "$N_PAR" -I{} bash -c \
  'args="{}"; name=$(echo "$args" | tr -d "-" | tr " " "_"); \
   python3 -m v9.run --exp train --skip-existing --preset '"$PRESET"' --episodes '"$EPISODES"' $args \
     > '"$LOGDIR"'/$name.log 2>&1; grep -E "TEST|skipping|Error" '"$LOGDIR"'/$name.log'

python3 -m v9.plots train "$PRESET"
