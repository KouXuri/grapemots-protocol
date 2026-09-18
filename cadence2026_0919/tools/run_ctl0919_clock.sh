#!/bin/bash
# Retention and motion-clock controls for the out-of-fold surface (review R-01).
#
# WHY. ultralytics 8.4.46 keeps a lost track for track_buffer PROCESSED frames and
# ignores the frame rate it is given (byte_tracker.py: max_frames_lost =
# args.track_buffer). A 30-frame buffer on the 15 Hz circling sequences is 2 s at
# Delta=1 and 16 s at Delta=8, and the Kalman filter advances one step per
# processed frame. So the Delta sweep moves retention time and the motion clock
# along with the number of observations. Three arms separate them, each at
# Delta in {2,4,8}, phase 0, all ten sequences, all five scales:
#   ret : retention held at about 2 s    buffer 15/8/4 for Delta 2/4/8
#   cu  : motion model on elapsed time   buffer 30, --catchup (Delta-1 extra predictions)
#   rt  : both                           buffer 15/8/4 and --catchup
# The Delta=1 cells need no rerun: at Delta=1 every arm is the 09-18 arm.
#
# PROVENANCE. tools/track_grapemots_mot.py (--catchup ported from
# tools/cadence_timescale_0815.py:catch_up), tracker configs differing from
# botsort_gmc.yaml in track_buffer only. Card 1 only.
set -u
PROJECT=/home/kou/my_env/yolo26; cd "$PROJECT"
PY="$PROJECT/.venv/bin/python"
Q=$PROJECT/runs/ctl_0919; mkdir -p "$Q/ret" "$Q/cu" "$Q/rt" "$Q/logs"
CARD=1
exec 9>"$Q/clock.lock"; flock -n 9 || { echo busy >&2; exit 2; }
log() { echo "[$(date '+%F %T')] $*" | tee -a "$Q/clock.log"; }
card_busy() { local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $CARD 2>/dev/null) || return 0; [ -n "$u" ] || return 0; [ "$u" -gt 1500 ]; }
MAP=$PROJECT/runs/grapemots_journal_phase2/results/lovo_weights_map.json
ROOT=$PROJECT/datasets/grapemots_det_721
VIDEOS="PathPlanning_2 PathPlanning_3 PathPlanning_4 PathPlanning_5 PathPlanning_6 PathPlanning_7 PathPlanning_8 NoPathPlanning_1 NoPathPlanning_2 NoPathPlanning_3"
buf() { case $1 in 2) echo 15;; 4) echo 8;; 8) echo 4;; esac; }
log "start"
for sz in 1536 2048 2560 3072 3840; do
  for dl in 8 4 2; do
    for arm in ret cu rt; do
      case $arm in
        ret) cfg=cfg/trackers/botsort_gmc_buf$(buf $dl).yaml; extra="";;
        cu)  cfg=cfg/trackers/botsort_gmc.yaml;                extra="--catchup";;
        rt)  cfg=cfg/trackers/botsort_gmc_buf$(buf $dl).yaml; extra="--catchup";;
      esac
      ok=1
      for v in $VIDEOS; do
        out="$Q/$arm/${v}_s${sz}_d${dl}.json"
        [ -f "$out" ] && continue
        w=$("$PY" -c "import json;print(json.load(open('$MAP'))['$v'])")
        while card_busy; do sleep 30; done
        env CUDA_VISIBLE_DEVICES=$CARD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" \
          tools/track_grapemots_mot.py --weights "$w" --root "$ROOT" --split all --videos "$v" \
          --tracker "$cfg" --detector-mode resize --imgsz "$sz" --global-imgsz "$sz" \
          --conf 0.25 --match-iou 0.5 --frame-step "$dl" $extra \
          --save-frame-boxes --save-frame-tracks --out "$out.tmp" \
          >"$Q/logs/${arm}_${v}_s${sz}_d${dl}.log" 2>&1 \
          && mv "$out.tmp" "$out" || { log "[FAIL] $arm $v s$sz d$dl"; ok=0; }
      done
      [ "$ok" = 1 ] && log "[OK] $arm sigma=$sz Delta=$dl" || log "[PARTIAL] $arm sigma=$sz Delta=$dl"
    done
  done
done
touch "$Q/clock.DONE"; log "[DONE] clock queue"
