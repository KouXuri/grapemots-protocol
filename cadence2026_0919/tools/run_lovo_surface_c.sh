#!/bin/bash
# The scale-cadence surface, out of fold, on ten sequences instead of two.
#
# WHY. Both CBDCom reviews asked about the level of inference, and review 1 asked
# specifically that sequence-level results be told apart from population claims.
# The rewritten headline was measured on the held-out pair, which is two
# sequences -- narrower than the evidence the reviews were reading. Ten
# leave-one-video-out checkpoints already exist, so the same surface can be read
# with every video scored by a checkpoint that never trained on it.
#
# PROVENANCE. Nothing new. tools/track_grapemots_mot.py unchanged, and
# --detector-mode resize is its own long-standing control detector; the weights
# map is runs/grapemots_journal_phase2/results/lovo_weights_map.json as it stands;
# the decomposition is tools/decompose_count_error.py. Only sigma, Delta and the
# video move. The one thing this script adds is the loop and the bookkeeping.
#
# Ordered cheapest first (cost goes as frames/Delta x ms(sigma)), every output
# file is its own mark, so an interrupted run resumes.
set -u
PROJECT=/home/kou/my_env/yolo26
cd "$PROJECT"
PY="$PROJECT/.venv/bin/python"
Q=$PROJECT/runs/lovo_surface_0918; mkdir -p "$Q/logs" "$Q/results"
exec 9>"$Q/queue.lock"; flock -n 9 || { echo busy >&2; exit 2; }
log() { echo "[$(date '+%F %T')] $*" | tee -a "$Q/queue.log"; }

SMI=/usr/bin/nvidia-smi
[ -x "$SMI" ] || SMI=$(command -v nvidia-smi || true)
card_busy() { local u; u=$("$SMI" --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null) || return 0; [ -n "$u" ] || return 0; [ "$u" -gt 1500 ]; }
# Excluding own pid is not enough. A $(...) substitution forks a subshell whose
# /proc cmdline is identical to this script's, so the check that runs inside the
# substitution is matched by the check itself and the job waits for its own
# subshell forever -- which is how the first attempt at this sweep parked for
# seven minutes on idle cards. Excluding this script by NAME removes the parent
# and every subshell it forks at once.
peer_alive() {
  local me hits
  me=$(basename "$0")
  hits=$(pgrep -af "^bash tools/run_" 2>/dev/null | grep -v -- "$me" || true)
  [ -n "$hits" ] && return 0
  pgrep -f "^[^ ]*python[0-9.]* tools/run_(v6_selection_queue|jafr)" >/dev/null && return 0
  return 1
}
log "parked: waiting for peers and card 0"
while peer_alive || card_busy; do sleep 60; done
sleep 30
log "card clear"

MAP=$PROJECT/runs/grapemots_journal_phase2/results/lovo_weights_map.json
ROOT=$PROJECT/datasets/grapemots_det_721
VIDEOS="PathPlanning_2 PathPlanning_3 PathPlanning_4 PathPlanning_5 PathPlanning_6 PathPlanning_7 PathPlanning_8 NoPathPlanning_1 NoPathPlanning_2 NoPathPlanning_3"

cell() {  # sigma  delta
  local sz=$1 dl=$2 done_all=1
  for v in $VIDEOS; do
    local out="$Q/results/${v}_s${sz}_d${dl}.json"
    [ -f "$out" ] && continue
    local w; w=$("$PY" -c "import json;print(json.load(open('$MAP'))['$v'])")
    [ -f "$w" ] || { log "[FAIL] $v: no checkpoint"; done_all=0; continue; }
    while card_busy; do sleep 30; done
    env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" \
      tools/track_grapemots_mot.py \
      --weights "$w" --root "$ROOT" --split all --videos "$v" \
      --tracker cfg/trackers/botsort_gmc.yaml \
      --detector-mode resize --imgsz "$sz" --global-imgsz "$sz" \
      --conf 0.25 --match-iou 0.5 --frame-step "$dl" \
      --save-frame-boxes --save-frame-tracks \
      --out "$out" >"$Q/logs/${v}_s${sz}_d${dl}.log" 2>&1 \
      || { log "[FAIL] ${v} s${sz} d${dl}"; done_all=0; }
  done
  [ "$done_all" = 1 ] && log "[OK] cell sigma=${sz} Delta=${dl} (10 videos)" \
                      || log "[PARTIAL] cell sigma=${sz} Delta=${dl}"
}

for sz in 1536 2048 2560 3072 3840; do
  for dl in 8 4 2 1; do cell "$sz" "$dl"; done
  log "--- sigma=${sz} complete ---"
done

log "--- reading the surface off the ten out-of-fold sequences ---"
"$PY" - "$Q/results" <<'PYEOF' | tee -a "$Q/queue.log"
import json, os, sys, glob, collections
sys.path.insert(0, 'tools')
from decompose_count_error import decompose
res = sys.argv[1]
MULTI = [f"PathPlanning_{i}" for i in range(2, 9)]
FRONT = [f"NoPathPlanning_{i}" for i in range(1, 4)]
cells = collections.defaultdict(dict)
for f in glob.glob(os.path.join(res, "*_s*_d*.json")):
    base = os.path.basename(f)[:-5]
    video, sz, dl = base.rsplit("_s", 1)[0], *base.rsplit("_s", 1)[1].split("_d")
    cells[(int(sz), int(dl))][video] = f

def pool(paths):
    tot = dict(P=0, G=0, U=0, D=0, M=0)
    for p in paths:
        j = json.load(open(p))
        for v in j["videos"]:
            stub = {"video": v["video"], "frame_predicted_ids": v["frame_predicted_ids"],
                    "frame_predicted_boxes": v["frame_predicted_boxes"],
                    "frame_gt_ids": v["frame_gt_ids"], "frame_gt_boxes": v["frame_gt_boxes"]}
            one = decompose(stub, 0.5, 1)
            for k in tot: tot[k] += one[k]
    return tot

for name, group in (("multi-view, 7 sequences", MULTI), ("frontal, 3 sequences", FRONT)):
    print(f"\n{name}: count error e = (P-G)/G at tau=1, out of fold\n")
    print(f"  {'sigma':>5s} " + "".join(f"{'D=%d' % d:>12s}" for d in (1, 2, 4, 8)))
    for sz in (1536, 2048, 2560, 3072, 3840):
        row = f"  {sz:5d} "
        for dl in (1, 2, 4, 8):
            have = cells.get((sz, dl), {})
            paths = [have[v] for v in group if v in have]
            if len(paths) != len(group):
                row += f"{'--':>12s}"; continue
            t = pool(paths)
            row += f"{(t['P']-t['G'])/t['G']:+12.4f}"
        print(row)
    print(f"  {'':5s} " + "".join(f"{'':>12s}" for _ in range(4)))
    for sz in (1536, 2048, 2560, 3072, 3840):
        have1 = cells.get((sz, 1), {}); have8 = cells.get((sz, 8), {})
        if len(have1) < len(group) or len(have8) < len(group): continue
        a = pool([have1[v] for v in group if v in have1])
        b = pool([have8[v] for v in group if v in have8])
        print(f"  sigma={sz}: coverage {1-a['M']/a['G']:.3f} at D=1 -> {1-b['M']/b['G']:.3f} at D=8, "
              f"U {a['U']}->{b['U']}, D {a['D']}->{b['D']}, M {a['M']}->{b['M']}")
PYEOF
log "[DONE] out-of-fold surface finished"
