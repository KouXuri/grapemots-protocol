#!/bin/bash
# Mac baseline: the same tiled arms as the frozen server run, re-run on this stack, so the
# whole-frame row differs from its reference in the read mode alone.
# cbdcom2026_r3), videos (SHA256 = same manifest), frame map, tracker config and
# scoring as the frozen tiled arms; only the detector read mode moves: one
# letterboxed pass at imgsz 4096, the native width, so bunches keep their pixels.
# Run on the Mac (Apple GPU, MPS); one output per sequence so a stop can resume.
set -u
cd /Users/xurikou/Desktop/yolo26
W=runs/wholeframe_2021_1001
PY=.venv/bin/python
CFG=grapemots-protocol/cfg/trackers/botsort_gmc.yaml
run() {  # group  weights-map  videos...
  local group=$1 map=$2; shift 2
  for v in "$@"; do
    local out=$W/results/tiledmac_${group}_${v}.json
    [ -f "$out" ] && continue
    echo "[$(date '+%F %T')] START $group $v" >> $W/logs/queue_tiledmac.log
    $PY tools/fullrate_decompose.py --root datasets/protocol_ext/bodegas2023 \
      --video-root datasets/protocol_ext/bodegas2023/videos --videos "$v" \
      --weights runs/detect/bodegas_piazolo/bodegas_yolo26s/weights/best.pt \
      --weights-map "$map" --frame-map $W/inputs/frame_map.json \
      --arm src_buf30:$CFG:source --arm rel_buf30:$CFG:released \
      --detector-mode tiled --imgsz 1280 --device mps \
      --out "$out" > $W/logs/tiledmac_${group}_${v}.log 2>&1 \
      && echo "[$(date '+%F %T')] OK $group $v" >> $W/logs/queue_tiledmac.log \
      || echo "[$(date '+%F %T')] FAIL $group $v" >> $W/logs/queue_tiledmac.log
  done
}
F1=$W/inputs/fold1_weights_map.json; F2=$W/inputs/fold2_weights_map.json
WANT=${WANT:-unseen seen}   # which groups to run, in order
# model-unseen: six read by fold 1, eleven by fold 2 (as decomp_fold1_six / fold2_eleven)
[[ " $WANT " == *" unseen "* ]] && {
run unseen $F2 row_4.3_2 row_4.4_2 row_4.4_4 row_6.1_3 row_6.1_4 row_6.2_1 row_6.2_2 row_7.1_3 row_7.1_4 row_7.2_1 row_7.2_2
run unseen $F1 row_4.2_1 row_6.1_1 row_6.1_2 row_7.1_1 row_7.1_2 row_8_1
}
echo "[$(date '+%F %T')] unseen pass ended" >> $W/logs/queue_tiledmac.log
echo "[$(date '+%F %T')] all complete" >> $W/logs/queue_tiledmac.log
