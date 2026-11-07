#!/usr/bin/env bash
# Multi-area study: local-only vs FedAvg (actor-only / actor+critic) vs pooled, N_PAR in parallel.
#   bash v9/run_multiarea.sh
set -euo pipefail
SEEDS=${SEEDS:-"0 1 2"}
EPISODES=${EPISODES:-2500}
N_PAR=${N_PAR:-4}
LOGDIR=${LOGDIR:-results_v9/logs/multiarea}
mkdir -p "$LOGDIR"

python3 -m v9.run --exp multiarea_baselines > "$LOGDIR/baselines.log" 2>&1 &

jobs=()
for s in $SEEDS; do
  jobs+=("--mode fedavg --E 4 --fed-parts actor --seed $s")
  jobs+=("--mode local --seed $s")
  jobs+=("--mode pooled --seed $s")
  jobs+=("--mode fedavg --E 4 --fed-parts both --seed $s")
done

printf '%s\n' "${jobs[@]}" | xargs -P "$N_PAR" -I{} bash -c \
  'args="{}"; name=$(echo "$args" | tr -d "-" | tr " " "_"); \
   python3 -m v9.run --exp multiarea --skip-existing --episodes '"$EPISODES"' $args \
     > '"$LOGDIR"'/$name.log 2>&1; grep -E "TEST|skipping|Error" '"$LOGDIR"'/$name.log'
wait
cat "$LOGDIR/baselines.log"
