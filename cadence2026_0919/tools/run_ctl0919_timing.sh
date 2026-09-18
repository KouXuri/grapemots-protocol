#!/bin/bash
# Same-session re-timing of the held-out-pair surface (review R-05).
#
# WHY. The published cost column mixed three dates (08-19, 08-25, 09-06) and
# divided by processed frames while calling it per source frame. This queue
# re-times every full-frame arm and the tiled arm at every interval in ONE
# session, three repeats, with nothing else on either card, and stores the
# per-frame timings so the lazy-initialisation frame can be dropped.
# Cost per annotated frame = sum(frame_ms) / annotated frames.
#
# Waits for runs/ctl_0919/{phase,clock}.DONE (files, not process names) and for
# both cards to be idle. Repeat 1 keeps boxes so the tiled Delta=2 arm, never run
# before, can be scored; repeats 2-3 keep ids only.
set -u
PROJECT=/home/kou/my_env/yolo26; cd "$PROJECT"
PY="$PROJECT/.venv/bin/python"
Q=$PROJECT/runs/ctl_0919; OUT=$Q/timing; mkdir -p "$OUT" "$Q/logs"
exec 9>"$Q/timing.lock"; flock -n 9 || { echo busy >&2; exit 2; }
log() { echo "[$(date '+%F %T')] $*" | tee -a "$Q/timing.log"; }
any_busy() { local u; for c in 0 1; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c 2>/dev/null) || return 0; [ -n "$u" ] || return 0; [ "$u" -gt 1500 ] && return 0; done; return 1; }
W=$PROJECT/runs/detect/cbdcom2026/gm_ctrl_newsplit_oldcfg/weights/best.pt
ROOT=$PROJECT/datasets/grapemots_det_721
log "waiting for phase.DONE and clock.DONE"
while [ ! -f "$Q/phase.DONE" ] || [ ! -f "$Q/clock.DONE" ]; do sleep 120; done
while any_busy; do sleep 60; done
log "start"
for rep in 1 2 3; do
  save="--save-frame-tracks"; [ "$rep" = 1 ] && save="--save-frame-boxes --save-frame-tracks"
  for dl in 1 2 4 8; do
    for arm in full1536 full2048 full2560 full3072 full3840 tiled1280; do
      out="$OUT/${arm}_d${dl}_r${rep}.json"
      [ -f "$out" ] && continue
      if [ "$arm" = tiled1280 ]; then
        mode="--detector-mode tiled --imgsz 1280 --tile 1280 --stride 960 --global-imgsz 1280"
      else
        sz=${arm#full}; mode="--detector-mode resize --imgsz $sz --global-imgsz $sz"
      fi
      while any_busy; do sleep 30; done
      env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" \
        tools/track_grapemots_mot.py --weights "$W" --root "$ROOT" --split test \
        --tracker botsort.yaml $mode --conf 0.25 --match-iou 0.5 --frame-step "$dl" \
        $save --out "$out.tmp" >"$Q/logs/timing_${arm}_d${dl}_r${rep}.log" 2>&1 \
        && mv "$out.tmp" "$out" || log "[FAIL] $arm d$dl r$rep"
    done
  done
  log "[OK] repeat $rep"
done
touch "$Q/timing.DONE"; log "[DONE] timing queue"
