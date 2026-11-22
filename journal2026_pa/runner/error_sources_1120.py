#!/usr/bin/env python3
"""Where a tracking-based bunch count goes wrong: replay, error substitution, attribution.

Existing code reused unchanged
  - Ultralytics 8.4.46 BOTSORT and its GMC (sparseOptFlow), built by
    track_grapemots_mot.build_tracker with cfg/trackers/botsort_gmc.yaml, fed exactly as
    track_grapemots_mot.py feeds it (Boxes of [x1, y1, x2, y2, score, 0], the frame image).
  - dump_dets_0920.py dumps (boxes and scores after resize_raw + merge_detections, i.e. the
    boxes the main grid's tracker received), load_gt_tracks for the reference boxes.
  - decompose_count_error.decompose / frame_matches (IoU 0.5, SciPy Hungarian) for U, D, M.

New here
  1. GMC cache. sparseOptFlow ignores the detections and BYTETracker.update calls it once
     per update whenever an image is passed, so its warps depend only on the frame
     sequence. They are computed once per (video, interval, phase) with the library GMC
     object, inside the same try/except as the library, and replayed by replacing
     gmc.apply. Checked: replay from the cache reproduces the main grid and the oracle
     arm by arm (`verify`).
  2. Error typing per processed frame, after TIDE (Bolya et al., ECCV 2020, Sec. 2.1-2.2,
     t_f = 0.5, t_b = 0.1): one-to-one matching at IoU >= 0.5 (frame_matches, the
     decomposition's Hungarian rule) gives true positives; an unmatched detection, taken
     highest score first, is background if its IoU-max reference box overlaps it by
     < 0.1, otherwise a localisation error on that box, which the localisation oracle
     snaps to the box unless the box is already covered, in which case it is suppressed
     (TIDE's duplicate rule). A reference box covered by neither is missed.
  3. Five fixes, each applied to every processed frame: bkg (remove background boxes),
     loc (snap assigned loc boxes to their reference box, drop unassignable ones),
     jit (snap true-positive boxes to their reference box), miss (add the uncovered
     reference boxes, at the median true-positive score of the dump), score (set every
     score to 0.9, the oracle's). All 32 subsets are run; with all five the input is the
     reference set at 0.9, i.e. the oracle up to box order.
  4. Exact Shapley attribution (Shapley 1953) of each output (P, U, D, M, e) to the five
     fixes, over all 2^5 subsets: phi_i = sum_S |S|!(n-|S|-1)!/n! [v(S+i) - v(S)].
     The five phi_i sum to v(all) - v(none) by construction (checked).

    python tools/error_sources_1120.py gmc-cache  --videos ... --deltas 1 2 4 8
    python tools/error_sources_1120.py verify
    python tools/error_sources_1120.py substitute --dets DUMP.json --deltas 1 2 4 8 --out DIR
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics as st
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path("/home/kou/my_env/yolo26")
sys.path.insert(0, str(ROOT / "tools"))
from ultralytics.engine.results import Boxes  # noqa: E402
from decompose_count_error import decompose, frame_matches, iou_matrix  # noqa: E402
from track_grapemots_mot import build_tracker, load_gt_tracks  # noqa: E402

DATA = ROOT / "datasets/grapemots_det_721"
CFG = "cfg/trackers/botsort_gmc.yaml"
OUT = ROOT / "runs/error_sources_1120"
CACHE = OUT / "gmc_cache"
FACTORS = ("bkg", "loc", "jit", "miss", "score")
torch.set_num_threads(1)
cv2.setNumThreads(1)


def video_frames(video):
    return sorted(p for p in (DATA / "images/all").iterdir()
                  if p.name.split("__")[0] == video and p.suffix.lower() in {".png", ".jpg", ".jpeg"})


# ------------------------------------------------------------------ GMC cache
def mem_available_gib():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    return 0.0


def wait_for_memory(floor=2.5):
    """The disease training shares this machine and has priority: never press it."""
    while mem_available_gib() < floor:
        time.sleep(30)


def gmc_cache(video, delta, phase=0):
    dest = CACHE / f"{video}_d{delta}_o{phase}.npy"
    if dest.exists():
        return dest
    wait_for_memory()
    tracker = build_tracker(CFG, frame_rate=30)
    gmc = tracker.gmc
    warps = []
    for p in video_frames(video)[phase::delta]:
        img = cv2.imread(str(p))
        try:
            w = gmc.apply(img, None)
        except Exception:               # the library's own fallback
            w = np.eye(2, 3)
        warps.append(np.asarray(w, dtype=np.float64))
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.stem + ".tmp.npy")
    np.save(tmp, np.stack(warps))
    tmp.rename(dest)                    # never leaves a partial cache file
    return dest


class CachedGMC:
    def __init__(self, warps):
        self.warps, self.i = warps, 0

    def apply(self, img, dets=None):
        w = self.warps[self.i]
        self.i += 1
        return w.copy()

    def reset_params(self):
        self.i = 0


# ------------------------------------------------------------------ one tracker pass
def track(frames, warps, size, cfg=CFG):
    """frames: list of dicts with boxes, scores, gt_ids, gt_boxes, name; returns our schema."""
    tracker = build_tracker(cfg, frame_rate=30)
    tracker.gmc = CachedGMC(warps)
    h, w = size
    dummy = np.zeros((1, 1, 3), dtype=np.uint8)          # non-None, so update() calls gmc.apply
    rec = {k: [] for k in ("frame_names", "frame_gt_ids", "frame_gt_boxes",
                           "frame_predicted_ids", "frame_predicted_boxes")}
    assert len(frames) == len(warps), (len(frames), len(warps))
    for f in frames:
        b = np.asarray(f["boxes"], dtype=np.float32).reshape(-1, 4)
        s = np.asarray(f["scores"], dtype=np.float32).reshape(-1)
        data = (np.concatenate([b, s[:, None], np.zeros((len(b), 1), dtype=np.float32)], axis=1)
                if len(b) else np.empty((0, 6), dtype=np.float32))
        tracks = tracker.update(Boxes(torch.as_tensor(data, dtype=torch.float32), (h, w)), dummy)
        rec["frame_names"].append(f["name"])
        rec["frame_gt_ids"].append([int(i) for i in f["gt_ids"]])
        rec["frame_gt_boxes"].append([[float(x) for x in bb] for bb in f["gt_boxes"]])
        if len(tracks):
            rec["frame_predicted_boxes"].append(np.asarray(tracks[:, :4], dtype=np.float32).tolist())
            rec["frame_predicted_ids"].append([int(t) for t in tracks[:, 4]])
        else:
            rec["frame_predicted_boxes"].append([])
            rec["frame_predicted_ids"].append([])
    return rec


# ------------------------------------------------------------------ error typing and fixes
def type_frame(boxes, scores, gt_ids, gt_boxes):
    """TP pairs, loc assignments, loc leftovers, background, missed reference indices."""
    n, m = len(boxes), len(gt_boxes)
    if n == 0:
        return dict(tp={}, loc={}, loc_drop=[], bkg=[], miss=list(range(m)))
    if m == 0:
        return dict(tp={}, loc={}, loc_drop=[], bkg=list(range(n)), miss=[])
    pairs = frame_matches(boxes, list(range(n)), gt_boxes, list(range(m)), 0.5)
    tp = {d: g for d, g in pairs}
    covered = set(tp.values())
    iou = iou_matrix(np.asarray(boxes, float), np.asarray(gt_boxes, float))
    loc, loc_drop, bkg = {}, [], []
    rest = sorted((d for d in range(n) if d not in tp), key=lambda d: -scores[d])
    for d in rest:                                       # TIDE: judged against the IoU-max reference box
        g = int(np.argmax(iou[d]))
        if iou[d, g] < 0.1:
            bkg.append(d)
        elif g not in covered:                           # localisation oracle: snap to it
            loc[d] = g
            covered.add(g)
        else:                                            # would duplicate a covered box: suppress
            loc_drop.append(d)
    miss = [g for g in range(m) if g not in covered]
    return dict(tp=tp, loc=loc, loc_drop=loc_drop, bkg=bkg, miss=miss)


def apply_fixes(f, fixes, add_score):
    boxes, scores = [list(map(float, b)) for b in f["boxes"]], [float(s) for s in f["scores"]]
    gtb = [list(map(float, b)) for b in f["gt_boxes"]]
    t = type_frame(boxes, scores, f["gt_ids"], gtb)
    ob, os_ = [], []
    for d in range(len(boxes)):                          # original order is kept
        if d in t["tp"]:
            ob.append(gtb[t["tp"][d]] if "jit" in fixes else boxes[d]); os_.append(scores[d])
        elif d in t["loc"]:
            ob.append(gtb[t["loc"][d]] if "loc" in fixes else boxes[d]); os_.append(scores[d])
        elif d in t["loc_drop"]:
            if "loc" not in fixes:
                ob.append(boxes[d]); os_.append(scores[d])
        else:                                            # background
            if "bkg" not in fixes:
                ob.append(boxes[d]); os_.append(scores[d])
    if "miss" in fixes:
        for g in t["miss"]:
            ob.append(gtb[g]); os_.append(add_score)
    if "score" in fixes:
        os_ = [0.9] * len(os_)
    return dict(name=f["name"], boxes=ob, scores=os_, gt_ids=f["gt_ids"], gt_boxes=f["gt_boxes"]), t


def frame_stats(frames):
    tp = loc = drop = bkg = miss = ref = 0
    tps = []
    for f in frames:
        t = type_frame([list(map(float, b)) for b in f["boxes"]], [float(s) for s in f["scores"]],
                       f["gt_ids"], [list(map(float, b)) for b in f["gt_boxes"]])
        tp += len(t["tp"]); loc += len(t["loc"]); drop += len(t["loc_drop"]); bkg += len(t["bkg"])
        miss += len(t["miss"]); ref += len(f["gt_boxes"])
        tps += [float(f["scores"][d]) for d in t["tp"]]
    return dict(reference_boxes=ref, tp=tp, loc=loc, loc_extra=drop, bkg=bkg, miss=miss,
                recall_iou50=round(tp / ref, 4) if ref else None,
                median_tp_score=round(st.median(tps), 4) if tps else 0.5)


def shapley(values):
    """values: dict frozenset(fixes) -> number; exact Shapley over FACTORS."""
    n = len(FACTORS)
    phi = {}
    for i in FACTORS:
        others = [x for x in FACTORS if x != i]
        tot = 0.0
        for r in range(n):
            for S in itertools.combinations(others, r):
                wgt = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
                tot += wgt * (values[frozenset(S) | {i}] - values[frozenset(S)])
        phi[i] = tot
    return phi


# ------------------------------------------------------------------ commands
def cmd_gmc_cache(a):
    for v in a.videos:
        for d in a.deltas:
            t0 = time.time()
            p = gmc_cache(v, d)
            print(f"gmc {v} d{d}: {p.name} {time.time() - t0:.1f} s", flush=True)


def load_dump(path):
    d = json.loads(Path(path).read_text())
    return d, d["frames"]


def size_of(video):
    img = cv2.imread(str(video_frames(video)[0]))
    return img.shape[:2]


def cmd_verify(a):
    """Replay against the frozen main grid (seed-0 whole-frame) and the oracle, arm by arm."""
    rep = {"main_grid": {}, "oracle": {}}
    dumps = sorted((ROOT / "runs/wholeframe_strongsort_1119/dets").glob("PathPlanning_*_s*.json"))
    for dp in dumps:
        d, frames = load_dump(dp)
        v, s = d["video"], d["imgsz"]
        hw = size_of(v)
        for delta in (1, 2, 4, 8):
            warps = np.load(gmc_cache(v, delta))
            rec = track(frames[::delta], warps, hw)
            ref = json.loads((ROOT / f"runs/fullframe_pponly_0922/surface_seed0/{v}_s{s}_d{delta}.json")
                             .read_text())["videos"][0]
            same_names = ref["frame_names"] == rec["frame_names"]
            same_ids = ref["frame_predicted_ids"] == rec["frame_predicted_ids"]
            dmax = max((abs(x - y) for fa, fb in zip(ref["frame_predicted_boxes"], rec["frame_predicted_boxes"])
                        for ba, bb in zip(fa, fb) for x, y in zip(ba, bb)), default=0.0)
            rep["main_grid"][f"{v}|{s}|{delta}"] = dict(names=same_names, ids=same_ids, max_box_diff=dmax)
            print(f"verify main {v} s{s} d{delta}: names {same_names} ids {same_ids} box {dmax:.2e}", flush=True)
    for v in [f"NoPathPlanning_{i}" for i in range(1, 4)] + [f"PathPlanning_{i}" for i in range(1, 9)]:
        frames_all = video_frames(v)
        hw = size_of(v)
        for delta in (1, 2, 4, 8):
            fr = []
            for p in frames_all[::delta]:
                gi, gb = load_gt_tracks(DATA / "tracks/all" / f"{p.stem}.txt", hw[1], hw[0])
                gb = np.asarray(gb, np.float32).reshape(-1, 4)
                fr.append(dict(name=p.stem + ".PNG", boxes=gb.tolist(), scores=[0.9] * len(gb),
                               gt_ids=[int(i) for i in gi], gt_boxes=gb.tolist()))
            rec = track(fr, np.load(gmc_cache(v, delta)), hw)
            ref = json.loads((ROOT / f"runs/oracle_cadence_1119/botsort_gmc/{v}_d{delta}_o0.json")
                             .read_text())["videos"][0]
            same_ids = ref["frame_predicted_ids"] == rec["frame_predicted_ids"]
            rep["oracle"][f"{v}|{delta}"] = dict(ids=same_ids)
            print(f"verify oracle {v} d{delta}: ids {same_ids}", flush=True)
    (OUT / "verify.json").write_text(json.dumps(rep, indent=1))
    n_main = sum(x["ids"] and x["names"] for x in rep["main_grid"].values())
    n_orc = sum(x["ids"] for x in rep["oracle"].values())
    print(f"main grid identical: {n_main}/{len(rep['main_grid'])}; oracle identical: {n_orc}/{len(rep['oracle'])}")


def cmd_substitute(a):
    d, frames = load_dump(a.dets)
    v, s = d["video"], d["imgsz"]
    if d.get("conf", 0.25) < 0.25:                       # same input as the main grid
        for f in frames:
            keep = [i for i, sc in enumerate(f["scores"]) if sc >= 0.25]
            f["boxes"] = [f["boxes"][i] for i in keep]
            f["scores"] = [f["scores"][i] for i in keep]
    tag = a.tag or Path(a.dets).stem
    out = Path(a.out) / f"{tag}.json"
    if out.exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    hw = size_of(v)
    stats = frame_stats(frames)
    add_score = stats["median_tp_score"]
    G_full = len({g for f in frames for g in f["gt_ids"]})
    res = dict(video=v, imgsz=s, weights=d["weights"], add_score=add_score, detection_stats=stats,
               G_full=G_full, arms={})
    for delta in a.deltas:
        warps = np.load(gmc_cache(v, delta))
        sub = frames[::delta]
        vals = {}
        for r in range(len(FACTORS) + 1):
            for S in itertools.combinations(FACTORS, r):
                fixed = [apply_fixes(f, set(S), add_score)[0] for f in sub]
                rec = track(fixed, warps, hw)
                rec["video"] = v
                t = decompose(rec, 0.5, 1)
                Mf = t["M"] + G_full - t["G"]
                vals[frozenset(S)] = dict(P=t["P"], U=t["U"], D=t["D"], M=Mf, e=(t["P"] - G_full) / G_full)
                if not S or len(S) == len(FACTORS):
                    keep = out.with_name(f"{tag}_d{delta}_{'all' if S else 'none'}.json")
                    keep.write_text(json.dumps({"config": {"fixes": list(S), "interval": delta, "dets": str(a.dets)},
                                                "videos": [rec]}))
        phi = {q: shapley({k: x[q] for k, x in vals.items()}) for q in ("P", "U", "D", "M", "e")}
        for q in phi:
            gap = vals[frozenset(FACTORS)][q] - vals[frozenset()][q]
            assert abs(sum(phi[q].values()) - gap) < 1e-9, (q, phi[q], gap)
        res["arms"][str(delta)] = dict(values={"+".join(sorted(k)) or "none": x for k, x in vals.items()},
                                       shapley=phi)
        print(f"substitute {tag} d{delta}: none e={vals[frozenset()]['e']:+.3f} "
              f"all e={vals[frozenset(FACTORS)]['e']:+.3f}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))


def cmd_replay_config(a):
    """Mechanism control: the same dumps under another tracker config (e.g. fuse_score off),
    for no fix, the miss fix and all fixes. Values only; same decomposition."""
    d, frames = load_dump(a.dets)
    v, s = d["video"], d["imgsz"]
    if d.get("conf", 0.25) < 0.25:
        for f in frames:
            keep = [i for i, sc in enumerate(f["scores"]) if sc >= 0.25]
            f["boxes"] = [f["boxes"][i] for i in keep]
            f["scores"] = [f["scores"][i] for i in keep]
    out = Path(a.out) / f"{a.tag}.json"
    if out.exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    hw = size_of(v)
    add_score = frame_stats(frames)["median_tp_score"]
    G_full = len({g for f in frames for g in f["gt_ids"]})
    res = dict(video=v, imgsz=s, cfg=a.cfg, G_full=G_full, add_score=add_score, arms={})
    for delta in a.deltas:
        warps = np.load(gmc_cache(v, delta))
        vals = {}
        for S in [(), ("miss",), ("miss", "score"), FACTORS]:
            fixed = [apply_fixes(f, set(S), add_score)[0] for f in frames[::delta]]
            rec = track(fixed, warps, hw, cfg=a.cfg)
            rec["video"] = v
            t = decompose(rec, 0.5, 1)
            Mf = t["M"] + G_full - t["G"]
            vals["+".join(sorted(S)) or "none"] = dict(P=t["P"], U=t["U"], D=t["D"], M=Mf, e=(t["P"] - G_full) / G_full)
            if getattr(a, "keep", False) and S in ((), FACTORS):
                kp = out.with_name(f"{a.tag}_d{delta}_{'all' if S else 'none'}.json")
                kp.write_text(json.dumps({"config": {"cfg": a.cfg, "fixes": list(S), "interval": delta, "dets": str(a.dets)},
                                          "videos": [rec]}))
        res["arms"][str(delta)] = vals
        print(f"replay-config {a.tag} d{delta}: none e={vals['none']['e']:+.3f}", flush=True)
    out.write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("gmc-cache"); g.add_argument("--videos", nargs="+", required=True)
    g.add_argument("--deltas", type=int, nargs="+", default=[1, 2, 4, 8])
    sp.add_parser("verify")
    s = sp.add_parser("substitute"); s.add_argument("--dets", required=True)
    s.add_argument("--deltas", type=int, nargs="+", default=[1, 2, 4, 8])
    s.add_argument("--out", default=str(OUT / "substitution")); s.add_argument("--tag")
    r = sp.add_parser("replay-config"); r.add_argument("--dets", required=True); r.add_argument("--cfg", required=True)
    r.add_argument("--keep", action="store_true", help="also write per-frame tracks of no fix and all fixes")
    r.add_argument("--deltas", type=int, nargs="+", default=[1, 2, 4, 8])
    r.add_argument("--out", required=True); r.add_argument("--tag", required=True)
    a = ap.parse_args()
    {"gmc-cache": cmd_gmc_cache, "verify": cmd_verify, "substitute": cmd_substitute,
     "replay-config": cmd_replay_config}[a.cmd](a)


if __name__ == "__main__":
    main()
