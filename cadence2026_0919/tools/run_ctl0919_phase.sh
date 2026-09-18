#!/bin/bash
# Sampling-phase control for the out-of-fold scale-cadence surface (review R-02).
#
# WHY. The 09-18 surface (runs/lovo_surface_0918) processed all_frames[0::Delta]
# only, so every cell is one of Delta possible samplings. This queue runs the
# other Delta-1 phases, all_frames[k::Delta] for k = 1..Delta-1, on the same ten
# sequences, the same leave-one-video-out checkpoints, the same tracker config
# (cfg/trackers/botsort_gmc.yaml), the same five scales. Only the phase moves.
#
# PROVENANCE. tools/track_grapemots_mot.py with its new --frame-offset flag; at
# offset 0 it reproduces the 09-18 JSON bit for bit (checked 2026-09-19 on
# PathPlanning_2, sigma 1536, Delta 8). Card 0 only; card 1 runs the clock queue.
# Each output is written to .tmp and renamed, so a file's existence marks it done.
set -u
PROJECT=/home/kou/my_env/yolo26; cd "$PROJECT"
PY="$PROJECT/.venv/bin/python"
Q=$PROJECT/runs/ctl_0919; OUT=$Q/phase; mkdir -p "$OUT" "$Q/logs"
CARD=0
exec 9>"$Q/phase.lock"; flock -n 9 || { echo busy >&2; exit 2; }
log() { echo "[$(date '+%F %T')] $*" | tee -a "$Q/phase.log"; }
# Gate on the fact (memory in use on this card), never on process names.
card_busy() { local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $CARD 2>/dev/null) || return 0; [ -n "$u" ] || return 0; [ "$u" -gt 1500 ]; }
MAP=$PROJECT/runs/grapemots_journal_phase2/results/lovo_weights_map.json
ROOT=$PROJECT/datasets/grapemots_det_721
VIDEOS="PathPlanning_2 PathPlanning_3 PathPlanning_4 PathPlanning_5 PathPlanning_6 PathPlanning_7 PathPlanning_8 NoPathPlanning_1 NoPathPlanning_2 NoPathPlanning_3"
log "start"
for sz in 1536 2048 2560 3072 3840; do
  for dl in 8 4 2; do
    ok=1
    for ((k=1; k<dl; k++)); do
      for v in $VIDEOS; do
        out="$OUT/${v}_s${sz}_d${dl}_o${k}.json"
        [ -f "$out" ] && continue
        w=$("$PY" -c "import json;print(json.load(open('$MAP'))['$v'])")
        while card_busy; do sleep 30; done
        env CUDA_VISIBLE_DEVICES=$CARD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" \
          tools/track_grapemots_mot.py --weights "$w" --root "$ROOT" --split all --videos "$v" \
          --tracker cfg/trackers/botsort_gmc.yaml --detector-mode resize --imgsz "$sz" --global-imgsz "$sz" \
          --conf 0.25 --match-iou 0.5 --frame-step "$dl" --frame-offset "$k" \
          --save-frame-boxes --save-frame-tracks --out "$out.tmp" \
          >"$Q/logs/phase_${v}_s${sz}_d${dl}_o${k}.log" 2>&1 \
          && mv "$out.tmp" "$out" || { log "[FAIL] $v s$sz d$dl o$k"; ok=0; }
      done
    done
    [ "$ok" = 1 ] && log "[OK] sigma=$sz Delta=$dl, phases 1..$((dl-1))" || log "[PARTIAL] sigma=$sz Delta=$dl"
  done
done
touch "$Q/phase.DONE"; log "[DONE] phase queue"
