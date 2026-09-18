#!/bin/bash
# Complete the AppleMOT scale-interval grid (review R-04: the dashes in Table IV).
# Same checkpoint, root, split, tracker and flags as the stage-13 cells in
# runs/caout_stage13_20260829/pa/ (read from their config); only the five missing
# (sigma, Delta) cells are run. Waits for timing.DONE so it cannot touch the timing.
set -u
PROJECT=/home/kou/my_env/yolo26; cd "$PROJECT"
PY="$PROJECT/.venv/bin/python"
Q=$PROJECT/runs/ctl_0919; OUT=$Q/apple; mkdir -p "$OUT" "$Q/logs"
exec 9>"$Q/apple.lock"; flock -n 9 || { echo busy >&2; exit 2; }
log() { echo "[$(date '+%F %T')] $*" | tee -a "$Q/apple.log"; }
card_busy() { local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null) || return 0; [ -n "$u" ] || return 0; [ "$u" -gt 1500 ]; }
W=$PROJECT/runs/detect/caout_stage7_20260827/ours_theirsplit/weights/best.pt
ROOT=$PROJECT/datasets/applemots_det_theirsplit
log "waiting for timing.DONE"
while [ ! -f "$Q/timing.DONE" ]; do sleep 120; done
for cell in "640 4" "640 8" "960 4" "960 8" "1280 8"; do
  set -- $cell; sz=$1; dl=$2
  out="$OUT/apple_s${sz}_d${dl}.json"; [ -f "$out" ] && continue
  while card_busy; do sleep 30; done
  env CUDA_VISIBLE_DEVICES=0 "$PY" tools/track_grapemots_mot.py --weights "$W" --root "$ROOT" \
    --split test --tracker botsort.yaml --detector-mode resize --imgsz "$sz" --global-imgsz "$sz" \
    --conf 0.25 --match-iou 0.5 --frame-step "$dl" --save-frame-boxes --save-frame-tracks \
    --out "$out.tmp" >"$Q/logs/apple_s${sz}_d${dl}.log" 2>&1 && mv "$out.tmp" "$out" \
    && log "[OK] apple s$sz d$dl" || log "[FAIL] apple s$sz d$dl"
done
touch "$Q/apple.DONE"; log "[DONE] apple queue"
