#!/bin/bash
# Reproduce the HARPO ReDial result: standard test (4,975 cases), full
# 6,630-movie catalogue, all four modules (CHARM, STAR, BRIDGE, MAVEN), MAVEN
# fitted on validation:  R@1 8.64 / R@10 30.05 / R@50 49.99 / MRR 0.156.
#
#   DATA=/path/to/redial_data OUT=/path/to/out bash reproduce.sh
#   (detached: setsid nohup bash reproduce.sh > reproduce.log 2>&1 < /dev/null &)
#
# DATA  converted ReDial: sft_data.json, test_sft.json, movie_list.json (sha256-checked)
# OUT   checkpoints, results, logs                        [./harpo_out]
# M05   Qwen2.5-0.5B-Instruct, local dir or HF id         [Qwen/Qwen2.5-0.5B-Instruct]
# M7    Qwen2.5-7B-Instruct, local dir or HF id           [Qwen/Qwen2.5-7B-Instruct]
# GPU0 GPU1  80 GB GPUs; GPU1=GPU0 runs everything on one [0 1]
# PY    python with requirements.txt installed            [python]
# DRY=1 print the commands instead of running them
#
# Each stage is skipped once it has finished (OUT/logs/<stage>.done), so the
# script can simply be re-run after an interruption. About 9 h on 2x A100 80 GB.
# GPU non-determinism makes a rerun close to, not bit-identical with, the result.

set -o pipefail
cd "$(dirname "$0")" || exit 1
DATA=${DATA:?set DATA to the converted ReDial directory}
OUT=$(mkdir -p "${OUT:-./harpo_out}" && cd "${OUT:-./harpo_out}" && pwd)
M05=${M05:-Qwen/Qwen2.5-0.5B-Instruct}
M7=${M7:-Qwen/Qwen2.5-7B-Instruct}
GPU0=${GPU0:-0}; GPU1=${GPU1:-1}
PY=${PY:-python}
CK=$OUT/checkpoints; RES=$OUT/results; LOG=$OUT/logs; AG=$RES/agents
[ -n "$DRY" ] && LOG=$OUT/logs/dry_run          # never touches a real run's markers
mkdir -p "$CK" "$RES/bridge" "$AG" "$LOG"
export PYTHONUNBUFFERED=1 TQDM_MININTERVAL=60 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# The exact data the result was produced from (the converter uses an LLM, so it
# is not regenerated here).
check_data() {
  sum() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }
  while read -r want file; do
    [ "$(sum "$DATA/$file")" = "$want" ] || { echo "!! $DATA/$file differs from the data the result used"; return 1; }
  done <<'EOF'
fcc1dc6889a8ca74dfc34219fbf922766c97bb83f385ef19aa53eddca43dabed sft_data.json
73bf3cb74c90fa40311b83d03d97558a1a727c187a1f697b1e57e0667b26d63c test_sft.json
fbc96491832a774824871ff256f5111b927579cd89006469796ca8d2dcf2eadf movie_list.json
EOF
}

# stage NAME GPU command...: run once, log to OUT/logs/NAME.log, mark NAME.done.
stage() {
  local name=$1 gpu=$2; shift 2
  [ -f "$LOG/$name.done" ] && { echo "-- $name: done earlier, skipped"; return 0; }
  echo "== $name (GPU $gpu)  $(date)"
  if [ -n "$DRY" ]; then echo "   CUDA_VISIBLE_DEVICES=$gpu $*"; touch "$LOG/$name.done"; return 0; fi
  if CUDA_VISIBLE_DEVICES=$gpu "$@" > "$LOG/$name.log" 2>&1; then
    touch "$LOG/$name.done"
  else
    echo "!! $name failed, see $LOG/$name.log"; touch "$LOG/FAILED"; return 1
  fi
}
# after STAGE...: wait for stages run by the other lane.
after() {
  for s in "$@"; do
    while [ ! -f "$LOG/$s.done" ]; do
      [ -f "$LOG/FAILED" ] && return 1
      sleep 60
    done
  done
}

S5=(--val-fraction 0.05)
BACKBONE=(--train-size 0 --test-size 0 --catalog-size 100000 --seq-len 256 --batch-size 16
          --grad-accum 1 --catalog-refresh 250)

# Lane B, part 1 (GPU1): CHARM pass 1 is a 7B cross-encoder trained on the
# shortlists of a 0.5B backbone that never saw CHARM's conversations; BRIDGE
# profiles are written by the plain 7B model.
lane_b1() {
  stage backbone_0.5b "$GPU1" $PY scripts/run_experiment.py --data "$DATA" --model "$M05" \
      "${BACKBONE[@]}" --epochs 6 "${S5[@]}" --charm-fraction 0.25 \
      --output "$CK/half_charm" --results-dir "$RES/half_charm" &&
  stage charm_pass1 "$GPU1" $PY scripts/train_charm_ce.py --data "$DATA" \
      --backbone "$CK/half_charm/checkpoints/sft_final" --base-model "$M7" --dtype bfloat16 \
      "${S5[@]}" --charm-fraction 0.25 --top-k 50 --group 16 --batch-groups 8 --eval-top-k 50 \
      --eval-every 1000 --score-dialogues 32 --no-test --results-dir "$RES/charm_pass1" &&
  stage bridge "$GPU1" $PY scripts/bridge_profiles.py --model "$M7" \
      --catalog "$CK/half_charm/catalog.json" --out "$RES/bridge/profiles.json"
}

# Lane A (GPU0): 7B backbone with validation held out; cold-start item vectors
# (chosen on validation); retriever + LM agents over the top-200; STAR.
lane_a() {
  local CK7=$CK/7b_val/checkpoints/sft_final
  stage backbone_7b "$GPU0" $PY scripts/run_experiment.py --data "$DATA" --model "$M7" \
      "${BACKBONE[@]}" --epochs 5 "${S5[@]}" --charm-fraction 0.0 \
      --output "$CK/7b_val" --results-dir "$RES/7b_val" &&
  stage coldstart_probe "$GPU0" $PY scripts/coldstart_probe.py --checkpoint "$CK7" --data "$DATA" \
      "${S5[@]}" --charm-fraction 0.0 --profiles "$RES/bridge/profiles.json" --top 200 \
      --out "$AG/coldstart_probe.json" || return 1
  for split in test val; do
    stage agents_$split "$GPU0" $PY scripts/rerank_eval.py --checkpoint "$CK7" --data "$DATA" \
        --split $split --top-k 200 "${S5[@]}" --charm-fraction 0.0 \
        --coldstart "$AG/coldstart_probe.json" --profiles "$RES/bridge/profiles.json" \
        --results-dir "$AG" || return 1
  done
  after bridge || return 1
  stage star_test "$GPU0" $PY scripts/star_rerank.py --model "$M7" --data "$DATA" \
      --profiles "$RES/bridge/profiles.json" --top-m 20 --branches 3 --batch 48 \
      --max-new-tokens 32 "${S5[@]}" --charm-fraction 0.0 --split test \
      --raw "$AG/rerank_top200_raw.pt" --out "$AG/star_test.pt" &&
  stage star_val "$GPU0" $PY scripts/star_rerank.py --model "$M7" --data "$DATA" \
      --profiles "$RES/bridge/profiles.json" --top-m 20 --branches 3 --batch 48 \
      --max-new-tokens 32 "${S5[@]}" --charm-fraction 0.0 --split val \
      --raw "$AG/rerank_top200_val_raw.pt" --out "$AG/star_val.pt"
}

# Lane B, part 2 (GPU1): CHARM continued on the 7B retriever's own top-200 hard
# negatives, the CHARM agent, then MAVEN fitted on validation.
lane_b2() {
  after backbone_7b || return 1
  stage charm_pass2 "$GPU1" $PY scripts/train_charm_ce.py --data "$DATA" \
      --backbone "$CK/7b_val/checkpoints/sft_final" --base-model "$M7" --dtype bfloat16 \
      "${S5[@]}" --charm-fraction 0.0 --top-k 200 --eval-top-k 100 --group 16 --batch-groups 8 \
      --max-train-cases 20000 --eval-every 1250 --score-dialogues 32 \
      --init-adapter "$RES/charm_pass1/charm_ce_adapter" --no-test --results-dir "$RES/charm_pass2" || return 1
  local adapter=$RES/charm_pass2/charm_ce_adapter
  [ -d "$adapter" ] || [ -n "$DRY" ] || adapter=$RES/charm_pass1/charm_ce_adapter
  for split in val test; do
    local raw=$AG/rerank_top200_raw.pt
    [ $split = val ] && raw=$AG/rerank_top200_val_raw.pt
    after agents_$split || return 1
    stage charm_$split "$GPU1" $PY scripts/score_shortlist_ce.py --adapter "$adapter" \
        --base-model "$M7" --dtype bfloat16 --data "$DATA" --raw "$raw" --split $split \
        "${S5[@]}" --charm-fraction 0.0 --out "$AG/charm_$split.pt" || return 1
  done
  after star_test star_val || return 1
  stage maven "$GPU1" $PY scripts/maven_fuse.py --test-raw "$AG/rerank_top200_raw.pt" \
      --val-raw "$AG/rerank_top200_val_raw.pt" --test-ce "$AG/charm_test.pt" \
      --val-ce "$AG/charm_val.pt" --agent star="$AG/star_test.pt,$AG/star_val.pt" \
      --fit-on val --out "$RES/maven.json"
}

echo "HARPO reproduction  commit $(git rev-parse --short HEAD 2>/dev/null)  $(date)"
echo "DATA=$DATA  OUT=$OUT  M05=$M05  M7=$M7  GPUs=$GPU0,$GPU1"
[ -n "$DRY" ] && rm -f "$LOG"/*.done
rm -f "$LOG/FAILED"
check_data || { [ -n "$ALLOW_OTHER_DATA" ] || exit 1; }

if [ "$GPU0" = "$GPU1" ]; then
  lane_b1 && lane_a && lane_b2
else
  lane_a & a=$!
  { lane_b1 && lane_b2; } & b=$!
  wait $a; wait $b
fi
[ -n "$DRY" ] && { rm -f "$LOG"/*.done; exit 0; }
[ -f "$LOG/maven.done" ] || { echo "!! not finished; re-run to resume"; exit 1; }
echo; grep -E "^agents:|std R@1|^agent: |^static consensus |^MAVEN \(per-dialogue\)" "$LOG/maven.log"
echo "result: $RES/maven.json  (headline: MAVEN (per-dialogue), standard)"
