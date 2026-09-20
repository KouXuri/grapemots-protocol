#!/bin/bash
# One GPU worker, pinned by PCI bus id. Usage: run_rev0920_worker3.sh CARD LIST [LIST...]
# __CARD__ in a job line is replaced by this worker's card: Ultralytics overwrites
# CUDA_VISIBLE_DEVICES with the device index it is passed, so training must be told
# the card through its own argument.
# CUDA orders devices by capability unless told otherwise, so CUDA_VISIBLE_DEVICES=1
# was landing on the card nvidia-smi calls 0 and the second worker sat in its memory
# gate. CUDA_DEVICE_ORDER=PCI_BUS_ID makes the two numbering schemes agree.
# Claims jobs by atomic mkdir under runs/rev_0920/claims, so workers never collide.
# Waits for its card to be free of other users (memory > 1500 MiB) before each job.
set -u
P=/home/kou/my_env/yolo26; cd "$P"
Q=$P/runs/rev_0920; CARD=$1; shift
mkdir -p "$Q/claims" "$Q/done" "$Q/fail" "$Q/logs"
exec 9>"$Q/worker3_$CARD.lock"; flock -n 9 || { echo busy >&2; exit 2; }
log() { echo "[$(date '+%F %T')] card$CARD $*" | tee -a "$Q/queue.log"; }
busy() { local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$CARD" 2>/dev/null) || return 0; [ -n "$u" ] || return 0; [ "$u" -gt 1500 ]; }
for list in "$@"; do
  while IFS=$'\t' read -r job cmd; do
    [ -z "$job" ] && continue
    [ -f "$Q/done/$job" ] && continue
    mkdir "$Q/claims/$job" 2>/dev/null || continue
    while busy; do sleep 60; done
    cmd="${cmd//__CARD__/$CARD}"
    log "start $job"
    if env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$CARD" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True bash -c "$cmd" > "$Q/logs/$job.log" 2>&1; then
      touch "$Q/done/$job"; log "[OK] $job"
    else
      touch "$Q/fail/$job"; log "[FAIL] $job"
    fi
  done < "$list"
done
log "[DONE] worker"
