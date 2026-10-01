#!/bin/bash
# Sequential master: held-out whole frames, then the Mac tiled baseline, then the
# seen whole frames. At imgsz 4096 one YOLO26 attention product needs a single 4 GB
# buffer, so the MPS allocator gets a soft watermark (frees its cache past ~6.4 GB)
# and a hard cap of ~15 GB (1.0 refused the 4 GB buffer on a fresh process); these govern memory only, not arithmetic.
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.6 PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.4
cd /Users/xurikou/Desktop/yolo26
W=runs/wholeframe_2021_1001
WANT=unseen $W/run_wholeframe.sh
WANT=unseen $W/run_tiledmac.sh
WANT=seen $W/run_wholeframe.sh
echo "[$(date '+%F %T')] run_all finished" >> $W/logs/queue.log
