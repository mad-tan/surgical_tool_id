#!/usr/bin/env bash
# Run one ~150 s time slice of `scripts/cv.py --exp EXP` as 4 parallel
# single-thread shards (used in the 3-minute-per-command sandbox; on a normal
# machine or Colab just run `python scripts/cv.py --exp EXP`).
# usage: scripts/run_slices.sh EXP [budget_seconds] [n_shards]
cd "$(dirname "$0")/.." || exit 1
export PYTHONDONTWRITEBYTECODE=1
EXP=$1; BUDGET=${2:-145}; N=${3:-4}
mkdir -p runs/logs "runs/cv/$EXP"
rm -f "runs/cv/$EXP"/*.lock   # locks only live for one slice
for i in $(seq 0 $((N-1))); do
  python3 scripts/cv.py --exp "$EXP" --budget "$BUDGET" --claim --threads 1 >> "runs/logs/$EXP.shard$i.log" 2>&1 &
  sleep 1
done
wait
python3 scripts/cv.py --exp "$EXP" --budget 0 2>&1 | grep -v -i -e warn -e paused | tail -6
