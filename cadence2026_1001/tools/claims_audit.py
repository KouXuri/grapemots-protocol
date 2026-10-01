#!/usr/bin/env python3
"""Every number in the 1 October 2026 manuscript that the table check
(verify_cbdcom_paper_1001.py) does not already cover, recomputed from archived
files with stock Python. Each line states the claim as printed and asserts it.

    python3 cadence2026_1001/tools/claims_audit.py        # from the archive root

Claims withdrawn during the audit are listed at the end with the value that
ruled them out, so the record shows why the sentence is gone.
"""
import glob, json, statistics as st, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "cadence2026_0919/tools"), str(ROOT / "cadence2026_0813/tools")]
import definition_sensitivity_0815 as ds  # noqa: E402

R13 = ROOT / "cadence2026_0813/results"
R3 = ROOT / "cbdcom2026_r3/results"
R26 = ROOT / "cadence2026/results"
H = ROOT / "cadence2026_1001/results"
J = lambda p: json.loads(Path(p).read_text())
results, withdrawn = [], []

def check(claim, ok, got):
    results.append((claim, bool(ok), got))

def e_of(v):
    return (v["P"] - v["G"]) / v["G"]

# ---------------------------------------------------------------- the two arms
runs = []
for f in ("arms_fold1_six.json", "arms_fold2_eleven.json"):
    runs += J(R13 / "adaptive_0813" / f)["runs"]
arm = lambda a: {r["video"]: ds.per_video(r, 0.5, 1) for r in runs if r["arm"] == a}
rel, src = arm("rel"), arm("src")
check("all 17 model-unseen sequences rise", len(rel) == 17 and all(src[s]["P"] > rel[s]["P"] for s in rel),
      sum(src[s]["P"] > rel[s]["P"] for s in rel))
seen = defaultdict(dict)
for f in ("decomp_seen_a.json", "decomp_seen_b.json"):
    for r in J(R3 / f)["runs"]:
        if r["arm"] in ("rel_buf30", "src_buf30"):
            seen[r["video"]][r["arm"]] = ds.per_video(r, 0.5, 1)["P"]
check("the 11 seen sequences rise too (28 of 28)",
      len(seen) == 11 and all(v["src_buf30"] > v["rel_buf30"] for v in seen.values()), len(seen))

pool = lambda d: {k: sum(v[k] for v in d.values()) for k in ("P", "G", "U", "D", "M")}
pr, ps = pool(rel), pool(src)
check("pooled e -0.298 -> +0.965; coverage 0.35 -> 0.69",
      (round(e_of(pr), 3), round(e_of(ps), 3), round(1 - pr["M"] / pr["G"], 2), round(1 - ps["M"] / ps["G"], 2))
      == (-0.298, 0.965, 0.35, 0.69), (e_of(pr), e_of(ps)))
check("U 99->347, D 19->86, M 219->106; 428 identities added",
      (pr["U"], ps["U"], pr["D"], ps["D"], pr["M"], ps["M"], ps["P"] - pr["P"]) == (99, 347, 19, 86, 219, 106, 428),
      (pr, ps))

def owned(record):
    life = Counter(t for fr in record["frame_predicted_ids"] for t in fr)
    ov = defaultdict(Counter)
    for pb, pi, gb, gi in zip(record["frame_predicted_boxes"], record["frame_predicted_ids"],
                              record["frame_gt_boxes"], record["frame_gt_ids"]):
        live = [(b, t) for b, t in zip(pb, pi) if life[t] >= 1]
        for p, g in ds.frame_matches([b for b, _ in live], [t for _, t in live], gb, gi, 0.5) if live else []:
            ov[p][g] += 1
    return {min(g for g, n in c.items() if n == max(c.values())) for c in ov.values() if c}
R = {(r["video"], r["arm"]): r for r in runs}
gain = sum(len(owned(R[(v, "src")]) - owned(R[(v, "rel")])) for v in rel)
loss = sum(len(owned(R[(v, "rel")]) - owned(R[(v, "src")])) for v in rel)
check("113 more trajectories reached (114 new, one lost)", (gain, loss) == (114, 1), (gain, loss))

cb = J(R13 / "decomp_0812/cluster_bootstrap.json")["by_tau"]["1"]["flights"]
meds = sorted(round(v["median_delta"], 2) for v in cb.values())
check("four flights all positive, medians +0.79..+1.58",
      len(cb) == 4 and all(v["all_positive"] for v in cb.values()) and (meds[0], meds[-1]) == (0.79, 1.58), meds)

# ---------------------------------------------------------------- definitions
iou = J(R13 / "definition_0815/definition_sensitivity_iou.json")
g = iou["gate"]
cov = [(round(g[f"iou{t}_tau1_rel"]["assigned_fraction"], 2), round(g[f"iou{t}_tau1_src"]["assigned_fraction"], 2))
       for t in ("0.2", "0.3", "0.5", "0.7")]
same = all((g[f"iou{t}_tau1_{a}"]["P"], g[f"iou{t}_tau1_{a}"]["G"]) == (g[f"iou0.5_tau1_{a}"]["P"], g[f"iou0.5_tau1_{a}"]["G"])
           for t in ("0.2", "0.3", "0.7") for a in ("rel", "src"))
check("ownership IoU 0.2/0.3/0.5/0.7: same P, G, e; coverage 0.42/0.41/0.35/0.25 vs 0.75/0.74/0.69/0.53",
      same and cov == [(0.42, 0.75), (0.41, 0.74), (0.35, 0.69), (0.25, 0.53)], cov)
ts = iou["tau_symmetry"]
check("tau=3 symmetric reference: G 299 instead of 339, e -0.682 and -0.060",
      (ts["tau3_rel"]["G_symmetric"], ts["tau3_rel"]["G_asymmetric"], round(ts["tau3_rel"]["e_symmetric"], 3),
       round(ts["tau3_src"]["e_symmetric"], 3)) == (299, 339, -0.682, -0.060), ts["tau3_src"]["e_symmetric"])
im = J(R13 / "definition_0815/identity_metrics.json")["rows"]
check("IDF1 0.284 vs 0.421; switches 18 vs 101; mostly-lost 259 vs 139",
      (round(im["rel"]["IDF1"], 3), round(im["src"]["IDF1"], 3), im["rel"]["IDSW"], im["src"]["IDSW"],
       im["rel"]["ML"], im["src"]["ML"]) == (0.284, 0.421, 18, 101, 259, 139), im["rel"]["IDF1"])

# ---------------------------------------------------------------- controls
b1 = {}
for f in ("decomp_buf1_fold1.json", "decomp_buf1_fold2.json"):
    for r in J(R3 / f)["runs"]:
        b1[r["video"]] = e_of(ds.per_video(r, 0.5, 1))
d1 = st.median(e_of(src[s]) - b1[s] for s in b1)
check("buffer of one processed frame leaves +1.063 of the gap (exactly 1.0625)", len(b1) == 17 and d1 == 1.0625, d1)
tsum = J(R13 / "timescale_0815/timescale_summary.json")["pairs"]["rel_dt->src/tau1"]
check("elapsed-clock contrast +1.000", round(tsum["delta_median"], 3) == 1.0, tsum["delta_median"])
lows = {a: sum(r.get("detections_low", 0) for r in runs if r["arm"] == a) for a in ("rel_low", "src_low")}
check("low-score arms add 1,415 and 41,443 candidates", (lows["rel_low"], lows["src_low"]) == (1415, 41443), lows)
check("low-score arms leave every count unchanged",
      all(ds.per_video(r, 0.5, 1)["P"] == ds.per_video(R[(r["video"], r["arm"][:-4])], 0.5, 1)["P"]
          for r in runs if r["arm"] in ("rel_low", "src_low")), True)
ss = [J(p)["arms"]["floor 0.1"]["stages"]["second"] for p in sorted((R13 / "adaptive_0813").glob("second_stage_*.json"))]
check("second stage: five sequences, 1,155 offered, none accepted",
      len(ss) == 5 and sum(s["candidates_offered"] for s in ss) == 1155 and sum(s["assignments_returned"] for s in ss) == 0,
      [s["candidates_offered"] for s in ss])

# ---------------------------------------------------------------- alignment
al = {Path(f).stem: J(f) for f in glob.glob(str(H / "bodegas_alignment_audit/alignment/row_*.json"))}
mae = [m["full_mae"] for d in al.values() for m in d["mappings"].values()]
off = [(s, n, m["full_mae"]) for s, d in al.items() for n, m in d["mappings"].items() if m["full_mae"] > 5]
check("679 images, all but 19 within 5, median 2.2; the 19 at 14-42, one per sequence, 18 of them the second image",
      len(mae) == 679 and len(off) == 19 and round(st.median(mae), 1) == 2.2
      and (round(min(x for *_, x in off)), round(max(x for *_, x in off))) == (14, 42)
      and len({s for s, *_ in off}) == 19 and sum(n == "000001.png" for _, n, _ in off) == 18, len(off))
aligned = [s for s in J(R26 / "bodegas_alignment_all28.json")["sequences"]]
check("the 29th sequence is row_6.3", set(al) - set(aligned) == {"row_6.3"}, set(al) - set(aligned))
lab = src_frames = 0; gaps, med_gap, span = [], [], []
for s in aligned:
    d = al[s]; idx = sorted(m["source_index"] for m in d["mappings"].values() if m["labelled"])
    lab += len(idx); src_frames += d["source_frames"]; iv = [b - a for a, b in zip(idx, idx[1:])]
    gaps += iv; med_gap.append(st.median(iv)); span.append((idx[-1] - idx[0]) / d["fps"])
check("28 sequences: 642 labelled of 25,864 source frames (2.5%), median interval 36 frames (1.67 Hz)",
      (lab, src_frames, round(100 * lab / src_frames, 1), st.median(gaps), round(59.94 / 36, 2)) == (642, 25864, 2.5, 36, 1.67),
      (lab, src_frames, st.median(gaps)))
check("six sequences at a median interval of 60 frames or more; longest span 32.7 s",
      sum(m >= 60 for m in med_gap) == 6 and round(max(span), 1) == 32.7, (sum(m >= 60 for m in med_gap), max(span)))
geo21 = J(R13 / "ext_cadence_0813/geometry_bodegas2023.json")["by_step"]["1"]
struct = [r for r in J(R26 / "sequence_structure.json")["sequences"] if r.get("corpus") == "bodegas2023"]
check("released interval: median r over 29 sequences 4.30, every one past sqrt(2), median overlap zero",
      geo21["sequences"] == 29 and round(geo21["sequence_median_r"], 2) == 4.30
      and min(r["step_over_size_median"] for r in struct) > 2 ** 0.5 and geo21["sequence_median_iou"] == 0.0,
      min(r["step_over_size_median"] for r in struct))

# ---------------------------------------------------------------- GrapeMOTS
ap = J(R13 / "ap_lovo_0814/lovo_ap_summary.json")["groups"]["3840x2160"]
check("nine 4K videos: median AP50 0.377, AP50:95 0.122",
      (ap["videos"], ap["median_ap50"], ap["median_ap50_95"]) == (9, 0.377, 0.122), ap)
pp1 = J(R13 / "ap_lovo_0814/ap_PathPlanning_1.json")["overall"]["ap50"]
check("PathPlanning_1's held-out checkpoint: AP50 0.004", round(pp1, 3) == 0.004, pp1)
lad = J(R26 / "density_realpipeline.json")["pooled_tau1"]
check("ladder U 713->81, D 480->49, M 87->247, G 415->391",
      tuple(lad[k][t] for k in ("k1", "k32") for t in ("U", "D", "M", "G")) == (713, 480, 87, 415, 81, 49, 247, 391), None)
cross = J(R13 / "fig_cancellation_data.json")["crossings"]
check("three paths reach zero with 0.39-0.48 of the reference",
      (round(min(cross.values()), 2), round(max(cross.values()), 2)) == (0.39, 0.48), cross)
pb = J(R13 / "decomp_0812/hota_panelB.json")["rows"]
low = min(pb, key=lambda k: pb[k]["signed_error"])
check("eleven videos, six configurations: smallest error with lowest coverage and HOTA",
      len(pb) == 6 and low == min(pb, key=lambda k: pb[k]["assigned_fraction"]) == min(pb, key=lambda k: pb[k]["HOTA"]), low)
assoc = J(R3 / "hota_assoc_rows.json")
assoc = assoc.get("rows", assoc)
check("ByteTrack with a 10-frame buffer: +3.783", round(assoc["ByteTrack, buffer 10"]["signed_error"], 3) == 3.783, None)

geo = J(R13 / "ext_cadence_0813/geometry_grapemots.json")["by_step"]
q = geo["2"]["r_earlier_box"]
check("tightest-band step on GrapeMOTS: median r 0.180, third quartile 0.313",
      (round(q["median"], 3), round(q["q3"], 3)) == (0.180, 0.313), q)
steps = [int(k) for k in geo if int(k) > 1]
xs, ys = steps, [geo[str(k)]["sequence_median_r"] for k in steps]
mx, my = st.mean(xs), st.mean(ys)
slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
r2 = 1 - sum((y - my - slope * (x - mx)) ** 2 for x, y in zip(xs, ys)) / sum((y - my) ** 2 for y in ys)
check("r linear in the interval along the ladder, R^2 > 0.999", r2 > 0.999, r2)
si = J(R3 / "scale_invariance.json")["2024 frontal"]
check("size slope on the frontal passes -0.82 over 45,819 pairs",
      round(si["log_slope"], 2) == -0.82 and sum(b[-1] for b in si["bins"]) == 45819, si["log_slope"])
cal = lambda pat: st.median(J(f)["c_per_second"] for f in glob.glob(str(pat)))
c_mv = cal(R26 / "calibration_PathPlanning_*.json")
c_fr = st.median(J(R26 / f"calibration_NoPathPlanning_{i}.json")["c_per_second"] for i in (1, 2, 3))
c_21 = cal(R3 / "calibration_2023/*.json")
check("c = 2.65 circling, 5.54 frontal, 6.38 on the 2021 flights",
      (round(c_mv, 2), round(c_fr, 2), round(c_21, 2)) == (2.65, 5.54, 6.38), (c_mv, c_fr, c_21))
band = lambda th, f, c: int(th * f / c)
check("Table IV steps", [band(t, 29.97, c_mv) for t in (.2, .4, .75)] == [2, 4, 8]
      and [band(t, 29.97, c_fr) for t in (.2, .4, .75)] == [1, 2, 4]
      and [band(t, 59.94, c_21) for t in (.2, .4, .75)] == [1, 3, 7], None)
cb2 = J(R13 / "c_bias_0814/c_estimator_bias.json")["sequences"]
check("lost trajectories 1.9 and 2.4 times faster; tracker's own c low by 18% and 59%",
      sorted(round(s["lost_over_kept"], 1) for s in cb2) == [1.9, 2.4]
      and sorted(round(100 * (1 - s["c_from_tracker_output"] / s["c_all_reference"])) for s in cb2) == [18, 59], None)
ph = J(R3 / "pilot_holdout.json")["pooled"]
check("one-second pilots: median |error| 41%", round(100 * ph["abs_rel_error_median"]) == 41, ph["abs_rel_error_median"])

# Fig. 3: crossings and surplus
dec = J(R13 / "decomp_0812/cadence_decomposition.json")["decomposition"]
def crossing(rs, es):
    import math
    for (r0, e0), (r1, e1) in zip(zip(rs, es), zip(rs[1:], es[1:])):
        if e0 > 0 >= e1:
            t = e0 / (e0 - e1); return math.exp(math.log(r0) + t * (math.log(r1) - math.log(r0)))
r21 = geo21["sequence_median_r"]
c21x = crossing([r21 / 36, r21], [dec["src_buf30"]["signed_error"], dec["rel_buf30"]["signed_error"]])
ks = [1, 2, 4, 8, 16, 32]
cgx = crossing([geo["2"]["sequence_median_r"] * k for k in ks], [lad[f"k{k}"]["signed_error"] for k in ks])
s21 = (dec["src_buf30"]["U"] + dec["src_buf30"]["D"]) / dec["src_buf30"]["G"]
sgm = (lad["k1"]["U"] + lad["k1"]["D"]) / lad["k1"]["G"]

# ---------------------------------------------------------------- frame budget, edge
pooled = lambda a: e_of(pool(arm(a)))
frames = Counter()
for r in runs:
    frames[r["arm"]] += r["tracker_frames"]
check("at 715 frames: frame difference +0.271, uniform -0.094; both reach more with frame difference",
      frames["uni2"] == frames["ada2"] == 715 and round(pooled("ada2"), 3) == 0.271 and round(pooled("uni2"), 3) == -0.094
      and all(pool(arm(f"ada{b}"))["M"] < pool(arm(f"uni{b}"))["M"] and pooled(f"ada{b}") > pooled(f"uni{b}") for b in (2, 4, 8)),
      (frames["uni2"], pooled("ada2"), pooled("uni2")))
edge = {Path(f).stem: J(f)["compute"] for f in glob.glob(str(R13 / "adaptive_0813/edge_*.json"))}
tr = sorted(v["stage_ms"]["track_ms_median"] for v in edge.values())
dets = sorted(v["detections_total"] for v in edge.values())
check("tracking step 78-84 ms whether 100 frames hold 1 detection or 1,859; about 12 fps",
      (round(tr[0]), round(tr[-1]), dets[0], dets[-1]) == (78, 84, 1, 1859) and round(1000 / tr[-1]) == 12, (tr, dets))
lk = J(R13 / "link_allintra_0814/link_allintra.json")
check("link: 97.2% of frames, 70% of bytes, 68.2 -> 20.3 Mbit/s, inter-coded 25.1",
      (round(100 * lk["frame_saving"], 1), round(100 * lk["byte_saving_same_codec"]), round(lk["fullrate_same_codec_mbit_s"], 1),
       round(lk["allintra_mbit_s_at_sparse_rate"], 1), round(lk["interframe_mbit_s_at_sparse_rate"] + 1e-9, 1))
      == (97.2, 70, 68.2, 20.3, 25.1), lk)

# ---------------------------------------------------------------- whole frames (2021)
wf = J(H / "wholeframe_2021.json")["whole frame, imgsz 4096 / unseen"]["tau=1"]
rl, sr = wf["pooled_rel"], wf["pooled_src"]
check("whole frame: count rises on all 17 sequences, paired +0.824, pooled -0.351 -> +0.552",
      (wf["up/down/tie"], f"{int(wf['delta_median'] * 1000 + 0.5) / 1000:+.3f}", f"{rl['e']:+.3f}", f"{sr['e']:+.3f}")
      == ("17/0/0", "+0.824", "-0.351", "+0.552"), wf["up/down/tie"])
check("whole frame: fewer tracks in both arms than the tiled read (220 < 238, 526 < 666)",
      rl["P"] < 238 and sr["P"] < 666, (rl["P"], sr["P"]))
check("whole frame: U and D grow faster than M shrinks (U 94->273, D 16->44, M 229->130)",
      (rl["U"], sr["U"], rl["D"], sr["D"], rl["M"], sr["M"]) == (94, 273, 16, 44, 229, 130)
      and sr["U"] - rl["U"] + sr["D"] - rl["D"] > rl["M"] - sr["M"], (rl, sr))
P = lambda runs: {r["arm"]: r["decomposition"]["1"]["P"] for r in runs if r["video"] == "row_4.3_2"
                  and r["arm"] in ("src_buf30", "rel_buf30")}
srv = P(J(R3 / "decomp_fold2_eleven.json")["runs"])
mac = P(J(H / "wholeframe_2021/repro_tiled_row_4.3_2_mps.json")["runs"])
cpu = P(J(H / "wholeframe_2021/repro_tiled_row_4.3_2_cpu.json")["runs"])
check("second machine, tiled re-run of row_4.3_2: 42 -> 40 and 19 -> 18 tracks (CPU and MPS agree)",
      (srv["src_buf30"], mac["src_buf30"], srv["rel_buf30"], mac["rel_buf30"]) == (42, 40, 19, 18) and mac == cpu,
      (srv, mac, cpu))

# ---------------------------------------------------------------- Track 3 review additions
check("pilots: 197 one-second pilots; prescribed step sparser than the held-out one in 48%",
      ph["pilots"] == 197 and round(100 * (1 - ph["conservative_fraction"])) == 48, ph)
uni, ada = pool(arm("uni2")), pool(arm("ada2"))
check("at 715 frames: coverage 0.49 (frame difference) against 0.45 (uniform)",
      (round(1 - ada["M"] / ada["G"], 2), round(1 - uni["M"] / uni["G"], 2)) == (0.49, 0.45), (ada, uni))
tiled_s = J(R13 / "adaptive_0813/edge_yolo26s_tiled.json")["compute"]
check("timing: decode 6 ms, eight-tile detection 197 ms, on CUDA, median of 100 frames",
      (round(tiled_s["stage_ms"]["decode_ms_median"]), round(tiled_s["stage_ms"]["detect_ms_median"]),
       tiled_s["frames"], tiled_s["device"].startswith("cuda")) == (6, 197, 100, True), tiled_s["stage_ms"])
check("bitrate: x264 CRF 23, rates per second of source footage (20.22 s, 1,212 frames)",
      (lk["crf"], lk["frames_total"], lk["step"]) == (23, 1212, 36), lk)
rc = J(H / "review_checks.json")
c21 = rc["2021_held_out"]
check("matched at least once: 0.354 / 0.690 against assigned 0.354 / 0.687; four multi-trajectory tracks",
      (c21["rel"]["contact_coverage"], c21["src"]["contact_coverage"], c21["rel"]["ownership_coverage"],
       round(c21["src"]["ownership_coverage"], 3), c21["rel"]["multi_trajectory_tracks"] + c21["src"]["multi_trajectory_tracks"])
      == (0.354, 0.6903, 0.354, 0.687, 4), c21)
lo = list(rc["2021_leave_one_flight_out"].values())
check("leave one flight out: all remaining rise; sparse -0.33..-0.28, source +0.79..+1.10",
      all(x["rises"] == x["sequences_left"] for x in lo)
      and (round(min(x["pooled_e_sparse"] for x in lo), 2), round(max(x["pooled_e_sparse"] for x in lo), 2),
           round(min(x["pooled_e_source"] for x in lo), 2), round(max(x["pooled_e_source"] for x in lo), 2))
      == (-0.33, -0.28, 0.79, 1.10), lo)
pv = list(rc["grapemots_per_video"].values())
check("GrapeMOTS per video: each rises in >= 14 of 15; median paired change +0.16..+0.51",
      min(x["rises"] for x in pv) == 14 and (round(min(x["median_delta_e"] for x in pv), 2),
                                              round(max(x["median_delta_e"] for x in pv), 2)) == (0.16, 0.51), pv)
cf = J(H / "conf_fill_identity.json")
check("Table III: confidence 0.75 IDF1 0.156 HOTA 0.152; 0.80 IDF1 0.090 HOTA 0.107",
      tuple(round(cf[k][m], 3) for k in ("Confidence 0.75", "Confidence 0.80") for m in ("IDF1", "HOTA"))
      == (0.156, 0.152, 0.090, 0.107), cf)

# ---------------------------------------------------------------- reference box size (pipeline paragraph)
boxes = [b for r in runs if r["arm"] == "rel" for f in r["frame_gt_boxes"] for b in f]
boxes += [b for f in ("decomp_seen_a.json", "decomp_seen_b.json") for r in J(R3 / f)["runs"]
          if r["arm"] == "rel_buf30" for fr in r["frame_gt_boxes"] for b in fr]
bw = st.median(x1 - x0 for x0, y0, x1, y1 in boxes); bh = st.median(y1 - y0 for x0, y0, x1, y1 in boxes)
check("median reference box 54 by 70 px on the 28 sequences; about 17 by 22 if 4096 is resized to 1280",
      (round(bw), round(bh), round(bw * 1280 / 4096), round(bh * 1280 / 4096)) == (54, 70, 17, 22), (bw, bh))

# ---------------------------------------------------------------- Fig. 3 (single panel)
ss = [s for s in J(R26 / "sequence_structure.json")["sequences"] if s.get("corpus") == "grapemots"]
check("GrapeMOTS at its labelling step: r 0.13-0.30, consecutive overlap 0.47-0.73 (11 sequences)",
      len(ss) == 11 and (round(min(s["step_over_size_median"] for s in ss), 2), round(max(s["step_over_size_median"] for s in ss), 2),
                         round(min(s["consecutive_iou_median"] for s in ss), 2), round(max(s["consecutive_iou_median"] for s in ss), 2))
      == (0.13, 0.30, 0.47, 0.73), None)
check("2021 arms in Fig. 2: uniform arms of 715, 1,403 and 2,779 frames between 371 and 12,386",
      (frames["rel"], frames["uni2"], frames["uni4"], frames["uni8"], frames["src"]) == (371, 715, 1403, 2779, 12386), frames)

# ---------------------------------------------------------------- withdrawn
near = {s: round(e_of(rel[s]), 3) for s in rel if abs(e_of(rel[s])) <= 0.10}
withdrawn.append(("'three sequences within 0.10 of zero at the released cadence'", near or "no such sequence"))
withdrawn.append(("'its own tracks return about half the true c'",
                  [round(s["c_from_tracker_output"] / s["c_all_reference"], 2) for s in cb2]))
withdrawn.append(("'crossing zero at half the frames' (an interpolation, replaced by the measured 715-frame pair)", None))
withdrawn.append(("'association rather than detection limits' (8-tile YOLO26s detects in 197 ms, tracks in 79 ms)", None))
withdrawn.append(("'the 2021 arms cross zero at r = 1.85' (interpolated between the two arms, 35 times apart in r; the panel was dropped)", None))

bad = 0
for claim, ok, got in results:
    bad += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {claim}" + ("" if ok else f"   got {got}"))
print("\nwithdrawn from the accepted text:")
for claim, why in withdrawn:
    print(f"  {claim}: {why}" if why is not None else f"  {claim}")
print(f"\n{len(results) - bad} of {len(results)} claims hold")
sys.exit(1 if bad else 0)
