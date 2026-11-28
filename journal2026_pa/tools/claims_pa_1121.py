#!/usr/bin/env python3
"""Claim-to-file table of the Precision Agriculture submission (manuscript_pa/main.tex), checked (2026-11-21).

Every number in the abstract, results, discussion, tables and captions is listed with the value printed in the
article, the value recomputed from the frozen result file, the file and key, and the producing tool. A row
passes when the printed value equals the frozen value at the printed precision (counts exactly).
Output: results/claims_pa_1121.csv and claims_pa_1121.md (in the results folder); exit code 1 if a row fails.
    .venv/bin/python cadence_journal_1119/tools/claims_pa_1121.py
"""
from __future__ import annotations

import csv
import json
import runpy
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"
J = lambda name: json.loads((RES / name).read_text())
O, N, S3, SA, SC = (J("oracle_trackers_summary_1120.json"), J("journal_numbers.json"),
                    J("error_sources_summary_s3072.json"), J("error_sources_summary.json"), J("story_checks_1120.json"))
CT, CR, GB, RT, RE = (J("calibration_transfer_1120.json"), J("calibration_robustness_1121.json"),
                      J("gate_buffer_summary_1121.json"), J("real_trackers_vine_1121.json"), J("reentry_analysis_1120.json"))
RM, SX = J("residual_motion_1120.json"), J("supplementary_checks_1121.json")
TRK = ("botsort", "bytetrack", "strongsort", "ocsort", "deepocsort")
SETS = ("crop_s0", "crop_s1", "crop_s2", "whole_s0", "whole_s1", "whole_s2")
VIDS = [f"PathPlanning_{i}" for i in range(2, 9)]
rows = []


def add(where, claim, printed, computed, ok, source, tool):
    rows.append(dict(where=where, claim=claim, printed=printed, computed=computed, ok="yes" if ok else "NO",
                     source=source, tool=tool))


from decimal import Decimal, ROUND_HALF_UP


def hu(x, nd):
    """round half up, as the article prints numbers"""
    q = Decimal(1).scaleb(-nd)
    v = Decimal(repr(x)).quantize(q, rounding=ROUND_HALF_UP)
    return int(v) if nd == 0 else float(v)


def pct(x, nd=0):
    return hu(100 * x, nd)


def r2(x):
    return hu(x, 2)


def ex(g):
    """exact count error of an oracle group from its counts"""
    return (g["P"] - g["G"]) / g["G"]


# ---------------------------------------------------------------- data
gf = N["G_full"]
add("Methods", "472 reference trajectories, 45 to 93 per vine", "472; 45-93",
    f"{sum(gf.values())}; {min(gf.values())}-{max(gf.values())}",
    sum(gf.values()) == 472 and min(gf.values()) == 45 and max(gf.values()) == 93,
    "journal_numbers.json: G_full", "journal_numbers.py")
fr = sum(RE["dataset"][v]["frames"] for v in VIDS)
add("Methods", "2,138 annotated frames on PP2-PP8", "2138", str(fr), fr == 2138,
    "reentry_analysis_1120.json: dataset[*].frames", "reentry_analysis_1120.py")
# ---------------------------------------------------------------- association alone
ds = RE["dataset"]
fr0 = [ds[f"NoPathPlanning_{i}"]["reappear_after_ge_2s"] for i in (1, 2, 3)]
re_ = [(ds[v]["reappear_after_ge_2s"], ds[v]["trajectories"]) for v in VIDS[2:]]
add("Results 1", "Re-entry after >=2 s: none on frontal passes; 2/93, 14/84, 18/51, 22/65, 42/77 on PP4-PP8 (none on PP2, PP3)",
    "0; 2/93 14/84 18/51 22/65 42/77", f"{sum(fr0)}; " + " ".join(f"{a}/{b}" for a, b in re_),
    sum(fr0) == 0 and re_ == [(2, 93), (14, 84), (18, 51), (22, 65), (42, 77)]
    and all(ds[v]["reappear_after_ge_2s"] == 0 for v in VIDS[:2]), "reentry_analysis_1120.json: dataset", "reentry_analysis_1120.py")
tot_re = sum(ds[v]["reappear_after_ge_2s"] for v in VIDS)
add("Results 1", "98 trajectories left the view for at least 2 s on PP2-PP8 (against 97-98 re-entry duplicates)", "98",
    str(tot_re), tot_re == 98, "reentry_analysis_1120.json: dataset[PP2-8].reappear_after_ge_2s", "reentry_analysis_1120.py")
mv = {k: O[k]["d1|tau1"]["groups"]["PP2-8"] for k in TRK}
fv = {k: O[k]["d2|tau1"]["groups"]["NP1-3"] for k in TRK}
ev = [pct(mv[k]["e"]) for k in TRK]
add("Abstract; Results 1; Fig. 1", "Oracle over-count of multi-view videos at 15 Hz: 66% (BoT-SORT) to 279% (ByteTrack)",
    "66-279%", f"{min(ev)}-{max(ev)}%", (min(ev), max(ev)) == (66, 279) and pct(mv['botsort']['e']) == 66 and pct(mv['bytetrack']['e']) == 279,
    "oracle_trackers_summary_1120.json: <tracker>/d1|tau1/groups/PP2-8/e", "oracle_trackers_summary_1120.py")
ratio = [mv[k]["e"] / fv[k]["e"] for k in TRK]
add("Abstract; Results 1", "Multi-view surplus 5.8 to 10 times the frontal one", "5.8-10",
    f"{min(ratio):.2f}-{max(ratio):.2f}", round(min(ratio), 1) == 5.8 and round(max(ratio)) == 10,
    "same file, PP2-8 e / NP1-3 e (d2)", "oracle_trackers_summary_1120.py")
fe = [pct(fv[k]["e"]) for k in TRK]
add("Discussion", "Frontal passes over-counted by 7% to 48% with perfect detection", "7-48%", f"{min(fe)}-{max(fe)}%",
    (min(fe), max(fe)) == (7, 48), "oracle_trackers_summary_1120.json: <tracker>/d2|tau1/groups/NP1-3/e", "oracle_trackers_summary_1120.py")
u = max(mv[k]["U_per_G"] for k in TRK); m = max(mv[k]["M_per_G"] for k in TRK)
add("Results 1", "Ownerless tracks at most 3.4% of the reference, unreached at most 2%", "3.4%; 2%",
    f"{100*u:.1f}%; {100*m:.1f}%", pct(u, 1) == 3.4 and pct(m) <= 2, "same file, U_per_G and M_per_G", "oracle_trackers_summary_1120.py")
reent = sorted({mv[k]["reentry_beyond_buffer"] for k in TRK})
add("Results 1", "Re-entries beyond retention 97 or 98 for every associator", "97-98", str(reent), reent == [97, 98],
    "same file, reentry_beyond_buffer", "oracle_trackers_summary_1120.py")
sw = {k: mv[k]["switch_in_view"] for k in TRK}
add("Results 1", "Switches in view 207-262 (motion-compensated), 868 OC-SORT, 1191 ByteTrack", "207-262; 868; 1191",
    f"{min(sw[k] for k in ('botsort','strongsort','deepocsort'))}-{max(sw[k] for k in ('botsort','strongsort','deepocsort'))}; {sw['ocsort']}; {sw['bytetrack']}",
    (min(sw[k] for k in ("botsort", "strongsort", "deepocsort")), max(sw[k] for k in ("botsort", "strongsort", "deepocsort")), sw["ocsort"], sw["bytetrack"]) == (207, 262, 868, 1191),
    "same file, switch_in_view", "oracle_trackers_summary_1120.py")
g1 = [RM[f"PathPlanning_{i}|d1"]["beyond_gate_after_gmc"] for i in range(2, 9)]
g2 = [RM[f"NoPathPlanning_{i}|d2"]["beyond_gate_after_gmc"] for i in (1, 2, 3)]
add("Results 1", "Compensated box fails the gate in 0.4-6.2% of steps (multi-view, 15 Hz), at most 0.7% frontal",
    "0.4-6.2%; 0.7%", f"{100*min(g1):.2f}-{100*max(g1):.2f}%; {100*max(g2):.2f}%",
    pct(min(g1), 1) == 0.4 and pct(max(g1), 1) == 6.2 and pct(max(g2), 1) <= 0.7, "residual_motion_1120.json: beyond_gate_after_gmc",
    "residual_motion_1120.py")
sp = [pct(ex(O["botsort"][f"d{d}|tau1"]["groups"]["PP2-8"])) for d in (2, 4, 8)]
add("Results 1", "BoT-SORT over-count rises to 117-137% at the sparser rates", "117-137%", f"{min(sp)}-{max(sp)}%",
    (min(sp), max(sp)) == (117, 137), "oracle_trackers_summary_1120.json: botsort/d2,d4,d8", "oracle_trackers_summary_1120.py")
h = GB
add("Results 1", "Relaxed gate: switches in view 207 -> 51 (-75%), over-count 66% -> 32%", "207->51; 75%; 66->32%",
    f"{h['H5a']['switch_in_view']['default']}->{h['H5a']['switch_in_view']['gate095']}; {100*h['H5a']['reduction']:.1f}%; {pct(h['default']['1']['oracle']['e'])}->{pct(h['gate095']['1']['oracle']['e'])}%",
    (h["H5a"]["switch_in_view"]["default"], h["H5a"]["switch_in_view"]["gate095"], pct(h["H5a"]["reduction"]), pct(h["default"]["1"]["oracle"]["e"]), pct(h["gate095"]["1"]["oracle"]["e"])) == (207, 51, 75, 66, 32),
    "gate_buffer_summary_1121.json: H5a, default/gate095 oracle e", "gate_buffer_summary_1121.py")
add("Results 1", "Fivefold retention: re-entries 98 -> 96", "98->96",
    f"{h['H5b']['reentry_beyond_buffer']['default']}->{h['H5b']['reentry_beyond_buffer']['buf150']}",
    (h["H5b"]["reentry_beyond_buffer"]["default"], h["H5b"]["reentry_beyond_buffer"]["buf150"]) == (98, 96),
    "gate_buffer_summary_1121.json: H5b", "gate_buffer_summary_1121.py")
d0, g0 = h["default"]["1"]["real"], h["gate095"]["1"]["real"]
add("Results 1", "Real detections, relaxed gate: total +50% -> -5%, U and D halved, M stays 55%, calibrated vine 46% -> 45%",
    "+50->-5%; 55%; 46->45%",
    f"{pct(d0['e']):+d}->{pct(g0['e']):+d}%; U {d0['U']}->{g0['U']}, D {d0['D']}->{g0['D']}; M {d0['M']}->{g0['M']}; {pct(d0['vine_lovo_median'])}->{pct(g0['vine_lovo_median'])}%",
    pct(d0["e"]) == 50 and pct(g0["e"]) == -5 and abs(g0["U"] / d0["U"] - 0.5) < 0.05 and abs(g0["D"] / d0["D"] - 0.5) < 0.05
    and pct(d0["M"]) == 55 and pct(g0["M"]) == 55 and pct(d0["vine_lovo_median"]) == 46 and pct(g0["vine_lovo_median"]) == 45,
    "gate_buffer_summary_1121.json: default/gate095 real at d1", "gate_buffer_summary_1121.py")
add("Fig. 1 caption", "G = 472 (PP2-PP8) and G = 256 (NP1-NP3)", "472; 256",
    f"{mv['botsort']['G']}; {fv['botsort']['G']}", (mv["botsort"]["G"], fv["botsort"]["G"]) == (472, 256),
    "oracle_trackers_summary_1120.json: groups/G", "oracle_trackers_summary_1120.py")
# ---------------------------------------------------------------- misses
rc = SX["recall_s3072"]
pooled = [rc[s]["pooled"] for s in SETS]
add("Results 2", "Detectors matched 22-36% of reference boxes in a frame at sigma 3072, 10-54% per video",
    "22-36%; 10-54%", f"{100*min(pooled):.1f}-{100*max(pooled):.1f}%; {100*min(rc[s]['per_video_min'] for s in SETS):.1f}-{100*max(rc[s]['per_video_max'] for s in SETS):.1f}%",
    (pct(min(pooled)), pct(max(pooled)), pct(min(rc[s]["per_video_min"] for s in SETS)), pct(max(rc[s]["per_video_max"] for s in SETS))) == (22, 36, 10, 54),
    "supplementary_checks_1121.json: recall_s3072", "supplementary_checks_1121.py")
w = SA["whole_s0"]["by_interval"]
sh1, sh8 = w["1"]["shapley"]["e"], w["8"]["shapley"]["e"]
add("Results 2; Table 1", "Seed-0 whole-frame, 15 Hz, pooled: e +0.50 = miss +1.10, confidence -0.50, background -0.23, localisation -0.21, jitter -0.01",
    "+0.50; +1.10 -0.50 -0.23 -0.21 -0.01", f"{w['1']['none']['e']:+.3f}; {sh1['miss']:+.3f} {sh1['score']:+.3f} {sh1['bkg']:+.3f} {sh1['loc']:+.3f} {sh1['jit']:+.3f}",
    (r2(w["1"]["none"]["e"]), r2(sh1["miss"]), r2(sh1["score"]), r2(sh1["bkg"]), r2(sh1["loc"]), r2(sh1["jit"])) == (0.50, 1.10, -0.50, -0.23, -0.21, -0.01),
    "error_sources_summary.json: whole_s0/by_interval/1", "error_sources_summary.py")
add("Table 1", "Delta=8 shares: background -0.04, localisation -0.03, jitter 0.00, miss +1.23, confidence +0.39; total +0.16 / +1.55",
    "-0.04 -0.03 0.00 +1.23 +0.39; +0.16 +1.55",
    f"{sh8['bkg']:+.3f} {sh8['loc']:+.3f} {sh8['jit']:+.3f} {sh8['miss']:+.3f} {sh8['score']:+.3f}; {sum(sh1.values()):+.3f} {sum(sh8.values()):+.3f}",
    (r2(sh8["bkg"]), r2(sh8["loc"]), abs(r2(sh8["jit"])), r2(sh8["miss"]), r2(sh8["score"]), r2(sum(sh1.values())), r2(sum(sh8.values()))) == (-0.04, -0.03, 0.0, 1.23, 0.39, 0.16, 1.55),
    "error_sources_summary.json: whole_s0/by_interval/8", "error_sources_summary.py")
tm = w["1"]["shapley"]
add("Table 1", "Terms moved at Delta=1: background U -0.21; localisation U -0.28; miss M -0.45, D +0.69 (0.695); confidence D -0.55",
    "-0.21; -0.28; -0.45, +0.69; -0.55", f"{tm['U']['bkg']}; {tm['U']['loc']}; {tm['M']['miss']}, {tm['D']['miss']}; {tm['D']['score']}",
    (r2(tm["U"]["bkg"]), r2(tm["U"]["loc"]), r2(tm["M"]["miss"]), r2(tm["D"]["score"])) == (-0.21, -0.28, -0.45, -0.55) and abs(tm["D"]["miss"] - 0.69) <= 0.0051,
    "error_sources_summary.json: whole_s0/by_interval/1/shapley/<U,D,M>", "error_sources_summary.py")
ms = list(SX["exact"]["shapley_miss_d1_sigma3072"].values())
add("Results 2", "Miss share over the six sets at sigma 3072, 15 Hz: +0.64 to +0.95", "+0.64..+0.95", f"{min(ms):+.3f}..{max(ms):+.3f}",
    (r2(min(ms)), r2(max(ms))) == (0.64, 0.95), "supplementary_checks_1121.json: exact/shapley_miss_d1_sigma3072 (error_sources_summary_s3072.json, rounded)", "supplementary_checks_1121.py")
worse, rng = 0, {}
MA = SX["exact"]["miss_alone_sigma3072"]
for d in ("1", "8"):
    e0 = [MA[d][s]["real"] for s in SETS]
    e1 = [MA[d][s]["miss_alone"] for s in SETS]
    rng[d] = (min(e0), max(e0), min(e1), max(e1))
for d in ("1", "2", "4", "8"):
    for s in SETS:
        b = S3[s]["by_interval"][d]
        worse += abs(b["none"]["e"] + b["alone"]["miss"]["e"]) > abs(b["none"]["e"])
add("Results 2; Fig. 2", "Missed boxes supplied alone: |e| larger in all 24 set x interval combinations", "24/24", f"{worse}/24", worse == 24,
    "error_sources_summary_s3072.json: none/e and alone/miss/e", "error_sources_summary.py")
add("Results 2", "15 Hz: +0.70..+1.67 -> +2.45..+3.11; 1.9 Hz: -0.31..+0.36 -> +0.68..+1.29",
    "+0.70..+1.67 -> +2.45..+3.11; -0.31..+0.36 -> +0.68..+1.29",
    "; ".join(f"{a:+.2f}..{b:+.2f} -> {c:+.2f}..{dd:+.2f}" for a, b, c, dd in (rng["1"], rng["8"])),
    tuple(map(r2, rng["1"])) == (0.70, 1.67, 2.45, 3.11) and tuple(map(r2, rng["8"])) == (-0.31, 0.36, 0.68, 1.29),
    "supplementary_checks_1121.json: exact/miss_alone_sigma3072 (error_sources_summary_s3072.json, rounded)", "supplementary_checks_1121.py")
add("Results 1; Fig. 2 dashed line", "Reference boxes leave the count 66% too high at 15 Hz and 117% at 1.9 Hz", "66%; 117%",
    f"{100*ex(O['botsort']['d1|tau1']['groups']['PP2-8']):.2f}%; {100*ex(O['botsort']['d8|tau1']['groups']['PP2-8']):.2f}%",
    (pct(ex(O["botsort"]["d1|tau1"]["groups"]["PP2-8"])), pct(ex(O["botsort"]["d8|tau1"]["groups"]["PP2-8"]))) == (66, 117),
    "oracle_trackers_summary_1120.json: botsort d1, d8", "oracle_trackers_summary_1120.py")
it_ = N["interval_steps_total"]
add("Results 2", "Count fell at the next sparser interval in 620 of 630 paired steps", "620/630", f"{it_['down']}/{it_['n']}",
    (it_["down"], it_["n"]) == (620, 630), "journal_numbers.json: interval_steps_total", "journal_numbers.py")
rv = SX["reversal"]
add("Results 2", "More rises than falls only with miss and confidence fixed, together with localisation or background",
    "condition", f"{rv['every_reversing_subset_has_miss_and_score']}, {rv['every_reversing_subset_also_fixes_loc_or_bkg']}, miss+score alone: {rv['miss_plus_score_alone_reverses_anywhere']}",
    rv["every_reversing_subset_has_miss_and_score"] and rv["every_reversing_subset_also_fixes_loc_or_bkg"] and not rv["miss_plus_score_alone_reverses_anywhere"],
    "supplementary_checks_1121.json: reversal", "supplementary_checks_1121.py")
H2 = SX["exact"]["seed_spread_sigma3072"]
add("Results 2", "Seed spread under the miss fix: 0.44 -> 0.28 (crop), 0.97 -> 0.49 (whole frame); less than halved",
    "0.44->0.28; 0.97->0.49",
    f"{H2['crop|none']['median_range']:.4f}->{H2['crop|miss']['median_range']:.4f}; {H2['whole|none']['median_range']:.4f}->{H2['whole|miss']['median_range']:.4f}",
    (r2(H2["crop|none"]["median_range"]), r2(H2["crop|miss"]["median_range"]), r2(H2["whole|none"]["median_range"]), r2(H2["whole|miss"]["median_range"])) == (0.44, 0.28, 0.97, 0.49)
    and H2["crop|miss"]["median_range"] > H2["crop|none"]["median_range"] / 2 and H2["whole|miss"]["median_range"] > H2["whole|none"]["median_range"] / 2,
    "supplementary_checks_1121.json: exact/seed_spread_sigma3072 (error_sources_summary_s3072.json h2, rounded)", "supplementary_checks_1121.py")
# ---------------------------------------------------------------- cancellation
nz, nz2 = SC["near_zero_arms"], N["near_zero"]
add("Results 3; Discussion", "103 of 840 arms within 10%; Gamma at least 0.98 (median 1.23); coverage at most 54%",
    "103/840; 0.98; 1.23; 54%", f"{nz['n']}/{nz2['arms']}; {nz['gross_min']}; {nz['gross_median']}; {100*nz2['max']:.1f}%",
    nz["n"] == 103 and nz2["arms"] == 840 and nz["gross_min"] >= 0.98 and r2(nz["gross_median"]) == 1.23 and pct(nz2["max"]) == 54,
    "story_checks_1120.json: near_zero_arms; journal_numbers.json: near_zero", "story_checks_1120.py, journal_numbers.py")
p1, p8 = SX["exact"]["pooled|whole|1"], SX["exact"]["pooled|whole|8"]
add("Results 3; Fig. 3", "Whole-frame pooled: U/G 0.79 -> 0.29, D/G 0.69 -> 0.29, M/G 0.44 -> 0.63, e +1.04 -> -0.05",
    "0.79->0.29; 0.69->0.29; 0.44->0.63; +1.04->-0.05",
    f"{p1['U_per_G']}->{p8['U_per_G']}; {p1['D_per_G']}->{p8['D_per_G']}; {p1['M_per_G']}->{p8['M_per_G']}; {p1['e']}->{p8['e']}",
    (r2(p1["U_per_G"]), r2(p8["U_per_G"]), r2(p1["D_per_G"]), r2(p8["D_per_G"]), r2(p1["M_per_G"]), r2(p8["M_per_G"]), r2(p1["e"]), r2(p8["e"])) == (0.79, 0.29, 0.69, 0.29, 0.44, 0.63, 1.04, -0.05),
    "supplementary_checks_1121.json: exact/pooled|whole|1,8 (unrounded; journal_numbers.json: pooled_terms_by_interval)", "journal_numbers.py")
add("Results 3", "Seed alone reversed the sign in 47/140 (crop) and 56/140 (whole frame) cells", "47/140; 56/140",
    f"{N['seed_effect_crop']['sign_flips']}/{N['seed_effect_crop']['cells']}; {N['seed_effect_whole']['sign_flips']}/{N['seed_effect_whole']['cells']}",
    (N["seed_effect_crop"]["sign_flips"], N["seed_effect_whole"]["sign_flips"], N["seed_effect_crop"]["cells"]) == (47, 56, 140),
    "journal_numbers.json: seed_effect_crop, seed_effect_whole", "journal_numbers.py")
aa = SC["aa_seed_pairs"]
add("Results 3", "Two checkpoints of one recipe differ by a median of 0.31, by >=1.09 (90th percentile) in a tenth of 840 pairs", "0.31; 1.09; 840",
    f"{aa['median']}; p90 {aa['p90']}; n {aa['n']}", r2(aa["median"]) == 0.31 and r2(aa["p90"]) == 1.09 and aa["n"] == 840,
    "story_checks_1120.json: aa_seed_pairs", "story_checks_1120.py")
vv = SX["validation_vs_count"]
add("Results 3", "Higher validation mAP had the larger |e| in 7/21 pairs at sigma 3072, 15 Hz; 8-10/21 under other summaries",
    "7/21; 8-10/21", ", ".join(f"{k} {v['higher_map_has_larger_error']}/{v['pairs']}" for k, v in vv.items()),
    vv["abs_e_sigma3072_15Hz"]["higher_map_has_larger_error"] == 7 and sorted(v["higher_map_has_larger_error"] for k, v in vv.items() if k != "abs_e_sigma3072_15Hz") == [8, 9, 10],
    "supplementary_checks_1121.json: validation_vs_count", "supplementary_checks_1121.py")
hv = N["hota_vs_count"]
add("Results 3", "Choosing the smallest count error gave up a median 42.5% of the attainable HOTA", "42.5%",
    f"{100*hv['median_hota_loss_choosing_count']:.1f}%", round(100 * hv["median_hota_loss_choosing_count"], 1) == 42.5,
    "journal_numbers.json: hota_vs_count", "journal_numbers.py")
# ---------------------------------------------------------------- vines
jt = CT["J_total_within_10pct"]
add("Results 4; Fig. 4a", "14 of 120 configurations within 10%, all mixing signs; median vine 29%, one in ten >82%",
    "14/120; 29%; 82%", f"{jt['configs']}/{len(CT['configs']) if isinstance(CT['configs'], list) else CT['configs']}; mixed {jt['configs_with_vines_of_both_signs']}; {jt['per_vine_abs_error']['median']}; p90 {jt['per_vine_abs_error']['p90']}",
    jt["configs"] == 14 and jt["configs_with_vines_of_both_signs"] == 14 and pct(jt["per_vine_abs_error"]["median"]) == 29 and pct(jt["per_vine_abs_error"]["p90"]) == 82,
    "calibration_transfer_1120.json: J_total_within_10pct", "calibration_transfer_1120.py")
tab = [("None", CT["A_lovo_uncalibrated"], (0.44, 1.78)), ("Ratio on the other six", CT["A_lovo_ratio"], (0.31, 1.20)),
       ("Straight line on the other six", CT["B_lovo_line"], (0.24, 0.82)),
       ("Best single ratio on all seven", CT["C_reference_same_config_per_vine"], (0.27, 0.97)),
       ("Ratio and configuration chosen on the other six", CR["nested_choice"]["per_vine"], (0.13, 1.33)),
       ("Carried to another seed (total)", CT["C_across_seed"]["pooled"], (0.19, 0.57)),
       ("Carried 15 Hz -> 1.9 Hz (total)", CT["D_across_interval_1_to_8"]["pooled"], (0.50, 0.59)),
       ("Carried to another scale (total)", CT["D_across_scale_from_3072"]["pooled"], (0.12, 0.47)),
       ("Carried to the other representation (total)", dict(CT["D_across_representation"]["pooled"], median=SX["exact"]["transfer_representation_pooled_median"]), (0.25, 0.51))]
for name, d, (med, p90) in tab:
    add("Table 2; Results 4", f"Calibration: {name} median / 90th percentile", f"{med:.2f} / {p90:.2f}", f"{d['median']} / {d['p90']}",
        (r2(d["median"]), r2(d["p90"])) == (med, p90), "calibration_transfer_1120.json / calibration_robustness_1121.json",
        "calibration_transfer_1120.py, calibration_robustness_1121.py")
eo = CT["E_oracle_botsort_phase0"]["1"]["lovo_ratio"]
add("Table 2", "Ratio on the other six, reference boxes at 15 Hz: 0.12 / 0.32", "0.12 / 0.32", f"{eo['median']} / {eo['p90']}",
    (r2(eo["median"]), r2(eo["p90"])) == (0.12, 0.32), "calibration_transfer_1120.json: E_oracle_botsort_phase0/1/lovo_ratio", "calibration_transfer_1120.py")
add("Results 4", "Seed transfer up to 98%; seed-averaged counts 27% (single seed 31%); median rho 0.32 (no configuration significant)",
    "98%; 27%; 0.32", f"{CT['C_across_seed']['pooled']['max']}; {CR['seed_ensemble_lovo']['median']}; {CT['F_spearman_P_vs_G']['median']}, share>=0.786 {CT['F_spearman_P_vs_G']['share_ge_0786']}",
    pct(CT["C_across_seed"]["pooled"]["max"]) == 98 and pct(CR["seed_ensemble_lovo"]["median"]) == 27 and pct(CR["single_seed_lovo_same_recipes"]["median"]) == 31 and r2(CT["F_spearman_P_vs_G"]["median"]) == 0.32 and CT["F_spearman_P_vs_G"]["share_ge_0786"] == 0,
    "calibration_transfer_1120.json, calibration_robustness_1121.json", "calibration_transfer_1120.py, calibration_robustness_1121.py")
# per-vine errors of the nested choice (PP8 -44%, PP6 x3.7, five within 14%)
NS = runpy.run_path(str(HERE / "tools/calibration_transfer_1120.py"), run_name="calib")
P, GF, VIDEOS = NS["P"], NS["GF"], NS["VIDEOS"]
chosen = {v: tuple(int(x) if x.isdigit() else x for x in s.split("|")) for v, s in CR["nested_choice"]["chosen"].items()}
err = {}
for v in VIDEOS:
    c = chosen[v]
    oth = [x for x in VIDEOS if x != v]
    k = sum(GF[x] for x in oth) / sum(P[(c[0], c[1], x, c[2], c[3])] for x in oth)
    err[v] = k * P[(c[0], c[1], v, c[2], c[3])] / GF[v] - 1
within = sorted(v[-1] for v in VIDEOS if abs(err[v]) <= 0.14)
add("Abstract; Results 4; Discussion", "Configuration and ratio chosen on the other six: five vines within 14%, PP8 -44%, PP6 at 3.7 x reference",
    "5 within 14%; -44%; 3.7x", f"{len(within)} ({','.join(within)}); {100*err['PathPlanning_8']:.1f}%; {1+err['PathPlanning_6']:.2f}x",
    len(within) == 5 and pct(err["PathPlanning_8"]) == -44 and round(1 + err["PathPlanning_6"], 1) == 3.7,
    "calibration_transfer_1120.py records + calibration_robustness_1121.json: nested_choice/chosen", "claims_pa_1121.py (recomputed)")
i1 = CT["I1_spearman_across_vines"]["UD_vs_M"]["median"]
add("Results 4", "Median rho between (U+D)/G and M/G over vines -0.79", "-0.79", str(i1), r2(i1) == -0.79,
    "calibration_transfer_1120.json: I1_spearman_across_vines/UD_vs_M", "calibration_transfer_1120.py")
fx = {s: J(f"calibration_under_fixes_{s}_s3072.json") for s in SETS}
nonef = [fx[s]["none"]["F"] for s in SETS]
add("Results 4", "sigma 3072: calibrated error 0.24 with reference boxes against 0.21-0.39 with real detections", "0.24; 0.21-0.39",
    f"{fx['whole_s0']['all']['F']}; {min(nonef)}-{max(nonef)}", r2(fx["whole_s0"]["all"]["F"]) == 0.24 and (r2(min(nonef)), r2(max(nonef))) == (0.21, 0.39),
    "calibration_under_fixes_<set>_s3072.json: none/F, all/F", "calibration_under_fixes_1120.py")
lv = {k: RT[k]["lovo_median"] for k in ("botsort", "strongsort", "ocsort", "deepocsort")}
oth = [lv[k] for k in ("strongsort", "ocsort", "deepocsort")]
add("Results 4", "Other associators on the same detections: median vine 48-56% off, against 55% for BoT-SORT", "48-56%; 55%",
    f"{100*min(oth):.1f}-{100*max(oth):.1f}%; {100*lv['botsort']:.1f}%", (pct(min(oth)), pct(max(oth)), pct(lv["botsort"])) == (48, 56, 55),
    "real_trackers_vine_1121.json: <tracker>/lovo_median", "real_trackers_vine_1121.py")

# ---------------------------------------------------------------- 2026-11-22 additions (ledger 10.28.16)
# no-video baseline: each vine predicted by the mean reference count of the other six
G7 = CT["G_full"]
base = [abs((sum(G7.values()) - g) / 6 - g) / g for g in G7.values()]
calc_q = NS["q"]
add("Results 4; Table 2", "Mean of the other six vines, no video: median 0.23, 90th percentile 0.46", "0.23; 0.46",
    f"{st.median(base):.4f}; {calc_q(base, 0.9):.4f}", (r2(st.median(base)), r2(calc_q(base, 0.9))) == (0.23, 0.46),
    "calibration_transfer_1120.json: G_full", "claims_pa_1121.py (recomputed, q() of calibration_transfer_1120.py)")
real_beats = [d for d in (CT["A_lovo_uncalibrated"], CT["A_lovo_ratio"], CT["B_lovo_line"])
              if d["median"] < st.median(base) and d["p90"] < calc_q(base, 0.9)]
eo1 = CT["E_oracle_botsort_phase0"]["1"]["lovo_ratio"]
add("Results 4; abstract; conclusions", "No real-detection calibration beats the baseline at both median and p90; the line "
    "stays above it in the median (0.24 against 0.23); reference boxes beat it on both", "line 0.24 vs 0.23; reference boxes 0.12/0.32",
    f"line {CT['B_lovo_line']['median']} vs {st.median(base):.4f}; ref {eo1['median']}/{eo1['p90']}; real beating both: {len(real_beats)}",
    not real_beats and (r2(CT["B_lovo_line"]["median"]), r2(st.median(base))) == (0.24, 0.23) and CT["B_lovo_line"]["median"] >= st.median(base)
    and eo1["median"] < st.median(base) and eo1["p90"] < calc_q(base, 0.9),
    "calibration_transfer_1120.json (the chosen configuration, 0.13/1.33, is checked above)", "claims_pa_1121.py (recomputed)")

# association alone and error substitution on APPLE MOTS and LettuceMOTS
EX = J("external_checks_1122.json")
TK = ("botsort_gmc", "bytetrack", "strongsort", "ocsort", "deepocsort")
og = lambda ds, t, grp: EX["oracle"][ds][f"{t}|d1|{grp}"]
for (ds, grp), printed in {("lettuce", "return"): [58, 90, 90, 57, 51], ("lettuce", "single"): [-1, 9, -1, -1, -1],
                            ("apple", "all"): [19, 72, 20, 37, 13]}.items():
    got = [pct(ex(og(ds, t, grp))) for t in TK]
    add("Table 3", f"Oracle count error (%), {ds} {grp}, BoT-SORT/ByteTrack/StrongSORT/OC-SORT/Deep OC-SORT", str(printed),
        str(got), got == printed, f"external_checks_1122.json: oracle/{ds}/<tracker>|d1|{grp}", "external_summary_1122.py")
Gs = (og("lettuce", "botsort_gmc", "return")["G"], og("lettuce", "botsort_gmc", "single")["G"], og("apple", "botsort_gmc", "all")["G"])
ab = lambda ds, vids: sum(EX["reference"][ds][v]["absent_ge30"] for v in vids)
absent = (ab("lettuce", EX["groups_lettuce"]["return"]), ab("lettuce", EX["groups_lettuce"]["single"]), ab("apple", EX["reference"]["apple"]))
add("Table 3; Results 5", "Trajectories 180/134/2,311; absent for >=30 frames 85/0/0", "180 134 2311; 85 0 0",
    f"{Gs}; {absent}", Gs == (180, 134, 2311) and absent == (85, 0, 0), "external_checks_1122.json: reference, oracle", "external_summary_1122.py")
ret = [ex(og("lettuce", t, "return")) for t in TK]
sing = [ex(og("lettuce", t, "single")) for t in TK]
app = [ex(og("apple", t, "all")) for t in TK]
add("Abstract", "Lettuce beds passed twice over-counted by 51-90% (single passes and apples: Table 3 rows)",
    "51-90", f"{100*min(ret):.2f}-{100*max(ret):.2f}", (pct(min(ret)), pct(max(ret))) == (51, 90),
    "external_checks_1122.json: oracle", "external_summary_1122.py")
reent = [og("lettuce", t, "return")["duplicate_origin"].get("reentry_beyond_buffer", 0) for t in TK]
shr = [r / og("lettuce", t, "return")["D"] for r, t in zip(reent, TK)]
sw = [og("apple", t, "all")["duplicate_origin"].get("switch_while_visible", 0) / og("apple", t, "all")["D"] for t in TK]
add("Results 5", "76-84 of the 85 returning plants got a new identity (52-89% of duplicates); apple switches in view 89-99% of duplicates",
    "76-84; 52-89%; 89-99%", f"{min(reent)}-{max(reent)}; {100*min(shr):.1f}-{100*max(shr):.1f}%; {100*min(sw):.1f}-{100*max(sw):.1f}%",
    (min(reent), max(reent)) == (76, 84) and (pct(min(shr)), pct(max(shr))) == (52, 89) and (pct(min(sw)), pct(max(sw))) == (89, 99),
    "external_checks_1122.json: oracle/<ds>/<tracker>|d1|<group>/duplicate_origin", "external_summary_1122.py (reentry_analysis_1120.py rule)")
SB = EX["substitution_apple"]
rec = [SB[f"s{s}|d1"]["tp"] / SB[f"s{s}|d1"]["reference_boxes"] for s in (640, 960, 1280)]
add("Results 5", "APPLE MOTS detector matched 15%, 45% and 68% of the reference boxes", "15 45 68",
    " ".join(f"{100*x:.2f}" for x in rec), [pct(x) for x in rec] == [15, 45, 68],
    "external_checks_1122.json: substitution_apple/<s>|d1/tp, reference_boxes", "external_summary_1122.py")
ee = {k: ((r["P_none"] - r["G"]) / r["G"], (r["P_miss"] - r["G"]) / r["G"], r["D_none"], r["D_miss"]) for k, r in SB.items()}
rose = [k for k, (a, b, _, _) in ee.items() if abs(b) > abs(a)]
flips = sorted((pct(a), pct(b)) for k, (a, b, _, _) in ee.items() if a * b < 0 and abs(abs(b) - abs(a)) < 0.03)
fell = [k for k, (a, b, _, _) in ee.items() if abs(b) < abs(a) and (pct(a), pct(b)) not in flips]
under = [ee[k][0] for k in fell]
add("Results 5", "Supplying misses added duplicates in all 12; |e| rose only at s1280/annotated rate (+7 -> +45); sign reversed in two "
    "(-37 -> +37, -21 -> +19); fell in nine that had under-counted by 50-94%", "12; s1280|d1 +7->+45; (-37,+37) (-21,+19); 9, 50-94",
    f"D up in {sum(d1 > d0 for _, _, d0, d1 in ee.values())}; rose {rose} {[(pct(ee[k][0]), pct(ee[k][1])) for k in rose]}; flips {flips}; "
    f"fell {len(fell)}, {100*max(under):.2f} to {100*min(under):.2f}",
    sum(d1 > d0 for _, _, d0, d1 in ee.values()) == 12 and rose == ["s1280|d1"] and (pct(ee["s1280|d1"][0]), pct(ee["s1280|d1"][1])) == (7, 45)
    and flips == [(-37, 37), (-21, 19)] and len(fell) == 9 and (pct(-max(under)), pct(-min(under))) == (50, 94),
    "external_checks_1122.json: substitution_apple", "external_summary_1122.py")

# AgriSORT (Lazio) grape sequences, queued by Codex as runs/pa_lazio_1122 (same five associators, reference boxes at 0.9)
LZ, LR = J("lazio_oracle_1122.json"), J("lazio_reference_1122.json")
lz = {t: dict(P=0, G=0) for t in TK}
for r in LZ["results"]:
    if r["interval"] == 1 and r["min_track_frames"] == 1:
        lz[r["tracker"]]["P"] += r["P"]; lz[r["tracker"]]["G"] += r["G"]
got = [pct(ex(lz[t])) for t in TK]
add("Table 3; Results 5", "AgriSORT count error (%) at the recorded rate, five associators; 96 bunches, none absent >=30 frames",
    "[4, 20, 8, 9, 8]; 96; 0", f"{got}; {lz['botsort_gmc']['G']}; {sum(v['absent_ge30'] for v in LR.values())}",
    got == [4, 20, 8, 9, 8] and lz["botsort_gmc"]["G"] == 96 and sum(v["absent_ge30"] for v in LR.values()) == 0
    and sum(v["frames"] for v in LR.values()) == 800,
    "lazio_oracle_1122.json: results (interval 1, min_track_frames 1); lazio_reference_1122.json", "queue_lazio_oracle_1122.py")

# vine-level calibration with independent detectors (runs/pa_vine_calibration_1122, 2026-11-23)
VC = J("vine_calibration_1122.json")
vm, vd = VC["metrics"], VC["design"]
fd = VC["folds"]
add("Methods", "Independent detectors: seven targets, seed 0, sigma 3072; PP1 + three test vines for training, one for "
    "checkpoint selection, two calibration vines", "7; 0; 3072; 4 (PP1+3); 1; 2",
    f"{vd['targets']}; {vd['seed']}; {vd['sigma']}; {vd['training_vines']}; {vd['validation_vines']}; {vd['calibration_vines']}",
    (vd["targets"], vd["seed"], vd["sigma"], vd["training_vines"], vd["validation_vines"], vd["calibration_vines"]) == (7, 0, 3072, 4, 1, 2)
    and all(f["training"][0] == "PathPlanning_1" and f["target"] not in f["training"] + [f["validation"]] + f["calibration"] for f in fd),
    "vine_calibration_1122.json: design, folds", "vine_calibration_1122.py")
for name, key, (med, p90) in (("Ratio on two vines, independent detector, 15 Hz", "ratio_d1", (0.34, 1.42)),
                              ("Ratio on two vines, independent detector, 1.9 Hz", "ratio_d8", (0.37, 1.14)),
                              ("Mean reference count of the two vines, no video", "mean_two", (0.34, 0.60))):
    d = vm[key]
    add("Table 2", f"{name}: median / 90th percentile", f"{med:.2f} / {p90:.2f}", f"{d['median']:.4f} / {d['p90']:.4f}",
        (r2(d["median"]), r2(d["p90"])) == (med, p90), f"vine_calibration_1122.json: metrics/{key}", "vine_calibration_1122.py")
no_better = all(vm[k]["median"] >= vm["mean_two"]["median"] and vm[k]["p90"] >= vm["mean_two"]["p90"] for k in ("ratio_d1", "ratio_d8"))
add("Results 4; Discussion", "Independent detectors: at 15 Hz and 1.9 Hz the two-vine ratio does no better than the two-vine mean "
    "(median and p90)", "no better at either rate",
    "; ".join(f"{k} {vm[k]['median']:.4f}/{vm[k]['p90']:.4f}" for k in ("ratio_d1", "ratio_d8", "mean_two")), no_better,
    "vine_calibration_1122.json: metrics", "vine_calibration_1122.py")
add("Results 4", "Same two calibration vines: cross-validated seed-0 counts 39% and 28%, independent detectors 34% and 37%",
    "39%, 28%; 34%, 37%", f"{vm['xval_ratio_d1']['median']:.4f}, {vm['xval_ratio_d8']['median']:.4f}; "
    f"{vm['ratio_d1']['median']:.4f}, {vm['ratio_d8']['median']:.4f}",
    (pct(vm["xval_ratio_d1"]["median"]), pct(vm["xval_ratio_d8"]["median"]), pct(vm["ratio_d1"]["median"]),
     pct(vm["ratio_d8"]["median"])) == (39, 28, 34, 37),
    "vine_calibration_1122.json: metrics/xval_ratio_d1, xval_ratio_d8, ratio_d1, ratio_d8", "vine_calibration_1122.py")

# ---------------------------------------------------------------- added after the pre-submission review (2026-11-25)
RC = J("review_checks_1125.json")
ns = RC["never_shown"]
add("Methods", "Trajectories absent from every processed frame: at most 4 of 472", "4; 472",
    f"{max(ns.values())} ({ns}); {RC['never_shown_of']}", max(ns.values()) == 4 and RC["never_shown_of"] == 472,
    "review_checks_1125.json: never_shown", "review_checks_1125.py")
add("Methods", "30 processed frames of retention = 2 s at 15 Hz and 16 s at 1.9 Hz (15/8 Hz)", "2; 16",
    f"{30 / 15:.3f}; {30 / (15 / 8):.3f}", (30 / 15, 30 / (15 / 8)) == (2.0, 16.0), "arithmetic: 30 frames / (15 Hz / interval)", "-")
DIB_T1 = dict(zip(VIDS, (209, 400, 373, 270, 284, 224, 374)))   # Ariza-Sentis et al. 2024, Data in Brief 54, Table 1
dsr = RE["dataset"]
rel = sum(dsr[v]["frames"] for v in VIDS)
add("Methods", "2,138 annotated frames in the release, 2,134 in the data paper's table", "2138; 2134",
    f"{rel}; {sum(DIB_T1.values())} (differ on PP" + ", PP".join(v[-1:] for v in VIDS if dsr[v]["frames"] != DIB_T1[v]) + ")",
    rel == 2138 and sum(DIB_T1.values()) == 2134, "reentry_analysis_1120.json: dataset/frames; data paper Table 1", "reentry_analysis_1120.py")
ra = RC["reentry_absences"]
add("Results 1", "71 of the 99 absences of 30 frames or more were shorter than the 150-frame retention", "71; 99",
    f"{ra['absences_shorter_than_150']}; {ra['absences_ge30']} ({ra['trajectories_with_absence_ge30']} trajectories)",
    (ra["absences_shorter_than_150"], ra["absences_ge30"], ra["trajectories_with_absence_ge30"]) == (71, 99, 98),
    "review_checks_1125.json: reentry_absences", "review_checks_1125.py")
ma = RC["miss_alone_per_vine"]
lo, hi = ma["exceptions_e_detected_range"]
add("Results 2", "Missed boxes supplied alone: |e| larger in 162 of 168 vine-level runs; the six exceptions under-counted by 38-86%",
    "162; 168; 38-86", f"{ma['worse']}; {ma['runs']}; {-100 * hi:.2f}-{-100 * lo:.2f}",
    (ma["worse"], ma["runs"], ma["better"], pct(-hi), pct(-lo)) == (162, 168, 6, 38, 86),
    "review_checks_1125.json: miss_alone_per_vine", "review_checks_1125.py")
rt_ = N["controls"]["ret"]
add("Results 2", "Retention scaled to about 2 s: seed-0 whole-frame count still fell in 101 of 105 steps", "101; 105",
    f"{rt_['down']}; {rt_['down'] + rt_['tie'] + rt_['up']}", (rt_["down"], rt_["down"] + rt_["tie"] + rt_["up"]) == (101, 105),
    "journal_numbers.json: controls/ret", "journal_numbers.py")
cc = RC["calibration_configs"]
add("Results 4", "Of 120 configurations, one beat the no-video baseline (median and p90) with the ratio, two with the line",
    "120; 1; 2", f"{cc['configurations']}; {len(cc['beat_ratio'])}; {len(cc['beat_line'])} "
    f"({', '.join(x['config'] for x in cc['beat_ratio'])} / {', '.join(x['config'] for x in cc['beat_line'])})",
    (cc["configurations"], len(cc["beat_ratio"]), len(cc["beat_line"])) == (120, 1, 2),
    "review_checks_1125.json: calibration_configs", "review_checks_1125.py")
iv = RC["independent_vines"]
add("Results 4", "Independent detectors beat the two-vine mean for three of the seven vines at 15 and at 1.9 Hz", "3; 3; 7",
    f"{iv['ratio_d1']}; {iv['ratio_d8']}; {RC['independent_vines_of']}",
    (iv["ratio_d1"], iv["ratio_d8"], RC["independent_vines_of"]) == (3, 3, 7), "review_checks_1125.json: independent_vines",
    "review_checks_1125.py")
n4 = [CT[k]["n"] for k in ("A_lovo_uncalibrated", "A_lovo_ratio", "B_lovo_line", "C_reference_same_config_per_vine")]
n6 = [CT["E_oracle_botsort_phase0"]["1"]["lovo_ratio"]["n"]] + [vm[k]["n"] for k in ("ratio_d1", "ratio_d8", "mean_two")]
nt = [CT[k]["pooled"]["n"] for k in ("C_across_seed", "D_across_interval_1_to_8", "D_across_scale_from_3072", "D_across_representation")]
add("Table 2 caption", "840 vine estimates in the first four rows, seven in the next six, 30-240 pairs for the totals",
    "840; 7; 30-240", f"{n4}; {n6}; {nt}", set(n4) == {840} and set(n6) == {7} and (min(nt), max(nt)) == (30, 240),
    "calibration_transfer_1120.json: n; vine_calibration_1122.json: n", "calibration_transfer_1120.py")

# ------------------------------------------------------------------ second pass of 25 November (ledger 10.28.17)
AS, RB = J("assa_1125.json"), J("robust_matching_1125.json")
nzA, allA, ob = AS["near_zero"]["AssA"], AS["all_arms"]["AssA"], AS["oracle_botsort"]
add("Results 3", "Arms within 10%: median AssA 0.22, at most 0.40; 0.22 over all arms; rho(AssA, |e|) -0.14 over 840 arms",
    "0.22; 0.40; 0.22; -0.14", f"{nzA['median']}; {nzA['max']}; {allA['median']}; {AS['spearman_AssA_vs_abs_e']} "
    f"(n {nzA['n']}/{allA['n']}; whole-frame check dev {AS['check_whole_vs_identity_metrics_max_abs_dev']})",
    (r2(nzA["median"]), r2(nzA["max"]), r2(allA["median"]), r2(AS["spearman_AssA_vs_abs_e"])) == (0.22, 0.40, 0.22, -0.14)
    and nzA["n"] == 103 and allA["n"] == 840 and AS["check_whole_vs_identity_metrics_max_abs_dev"] < 1e-9,
    "assa_1125.json: near_zero, all_arms, spearman_AssA_vs_abs_e", "assa_1125.py")
add("Results 1", "Oracle BoT-SORT: median AssA 0.71 at 15 Hz, 0.20 at 1.9 Hz", "0.71; 0.20",
    f"{ob['1']['AssA']['median']}; {ob['8']['AssA']['median']}",
    (r2(ob["1"]["AssA"]["median"]), r2(ob["8"]["AssA"]["median"])) == (0.71, 0.20),
    "assa_1125.json: oracle_botsort", "assa_1125.py")
gv = RB["gated_vs_default"]
alt = [RB[k] for k in ("default_0.3_tau1", "default_0.7_tau1", "default_0.5_tau3")]
add("Results 3", "Scoring rule: gated matching changes U, D or M by one in 4 of 840 arms; IoU 0.3/0.7 or tau 3 keep "
    "Gamma >= 0.82 and coverage <= 62% for the arms within 10%", "4; 1; 0.82; 62%",
    f"{gv['arms_changed']}; {gv['max_abs_change_in_U_D_or_M']}; {min(a['gross_min'] for a in alt)}; "
    f"{100 * max(a['coverage_max'] for a in alt):.1f}% (near-zero arms {[a['near_zero'] for a in alt]})",
    (gv["arms_changed"], gv["max_abs_change_in_U_D_or_M"]) == (4, 1) and min(a["gross_min"] for a in alt) >= 0.82
    and hu(100 * max(a["coverage_max"] for a in alt), 0) == 62 and RB["default_0.5_tau1"]["near_zero"] == 103,
    "robust_matching_1125.json", "robust_matching_1125.py")
rv = RC["recall_vs_error"]
add("Results 2", "Six detector sets: recall and e ranked alike at every interval (rho 0.83-1.00); at 15 Hz rho 1.00 "
    "with D, every set over-counted", "0.83-1.00; 1.00",
    f"{rv['rho_e']}; D {rv['rho_D_15Hz']}; M {rv['rho_M_15Hz']}; min e at 15 Hz {min(rv['e_15Hz'].values())}",
    (r2(min(rv["rho_e"].values())), r2(max(rv["rho_e"].values()))) == (0.83, 1.00) and r2(rv["rho_D_15Hz"]) == 1.00
    and min(rv["e_15Hz"].values()) > 0, "review_checks_1125.json: recall_vs_error", "review_checks_1125.py")
vb = RC["vine_bunches"]
add("Results 4", "Median vine off by 20 bunches (14 near-total configs), 21 after the ratio, 19 for the mean of the other six",
    "20; 21; 19", f"{vb['near_total']['median']}; {vb['ratio_other_six']['median']}; {vb['mean_other_six']['median']} "
    f"(n {vb['near_total']['n']}, {vb['ratio_other_six']['n']}, {vb['mean_other_six']['n']})",
    (hu(vb["near_total"]["median"], 0), hu(vb["ratio_other_six"]["median"], 0), hu(vb["mean_other_six"]["median"], 0)) == (20, 21, 19)
    and vb["near_total_configs"] == 14, "review_checks_1125.json: vine_bunches", "review_checks_1125.py")
ph = N["controls"]["phase"]
add("Results 2", "Seed-0 whole-frame: count fell at the next sparser interval in 103 of 105 steps for the lowest and the highest "
    "count over all starting frames", "103; 103; 105",
    f"{ph['steps_down_lowest']}; {ph['steps_down_highest']}; {ph['cells']}",
    (ph["steps_down_lowest"], ph["steps_down_highest"], ph["cells"]) == (103, 103, 105),
    "journal_numbers.json: controls/phase", "journal_numbers.py")

# ------------------------------------------------------------------ 27 November: L and R of the 24 November queue
FU = J("pa_followup_readout_1127.json")
lr, la, lb = FU["L"]["recall_iou50"], FU["L"]["H7a"], FU["L"]["H7b"]
hi = [v for k, v in lr.items() if k.endswith(("|816", "|1088"))]
e_over = [v["none"]["e"] for v in FU["L"]["rows"].values() if v["abs_e_up"] is not None]
add("Methods; Results 5", "LettuceMOTS real detector: 95-97% of plants found at 816/1088 px; +62% at 1088 px and 15 Hz against "
    "+68% with the reference boxes; 0.63 duplicates per plant; misses supplied raise |e| in all 11 over-counting "
    "combinations and add duplicates in 12 of 16", "95-97%; +62%; +68%; 0.63; 11/11; 12/16",
    f"{min(hi)}-{max(hi)}; {la['e_real']}; {la['e_reference_boxes']}; {la['D_per_G']}; {lb['abs_e_up']}/{lb['combos_e_nonneg']}; "
    f"{lb['D_up']}/{lb['combos']}",
    (hu(100 * min(hi), 0), hu(100 * max(hi), 0)) == (95, 97) and (pct(la["e_real"]), pct(la["e_reference_boxes"])) == (62, 68)
    and r2(la["D_per_G"]) == 0.63 and (lb["abs_e_up"], lb["combos_e_nonneg"], lb["D_up"], lb["combos"]) == (11, 11, 12, 16)
    and min(e_over) > 0, "pa_followup_readout_1127.json: L", "pa_followup_readout_1127.py")
rr = FU["R"]
add("Methods", "One crop checkpoint (PathPlanning_4_s2) first trained at batch 4 after an OOM was retrained at batch 8 and "
    "replaced the published one (median |delta e| 0.34 >= 0.30, the rule fixed before the run)", "8; replaced",
    f"batch {rr['trained']['batch_requested']}->{rr['trained']['batch_used']}; median |de| {rr['median_abs_delta']}; {rr['verdict']}",
    rr["trained"]["batch_used"] == 8 and rr["median_abs_delta"] >= 0.30 and rr["verdict"].startswith("replace")
    and all(x["replay"] == x["live"] for x in rr["pipeline_check_3072"].values()),
    "pa_followup_readout_1127.json: R; crop_batch8_1127.py", "pa_followup_readout_1127.py")

# ------------------------------------------------------------------ 28 November: M (YOLO26m) of the 24 November queue
mm = FU["M"]
mdd = mm["descriptive"]
assert mm["complete"], "M incomplete"
add("Methods", "YOLO26m: 21 checkpoints (3 seeds x 7 folds), all at batch 4 because batch 8 did not fit in GPU memory",
    "21; 4", f"{len(mm['batch_used'])}; {sorted(set(mm['batch_used'].values()))}",
    len(mm["batch_used"]) == 21 and set(mm["batch_used"].values()) == {4} and mm["seeds_complete"] == [0, 1, 2],
    "pa_followup_readout_1127.json: M/batch_used (trained.json)", "pa_followup_readout_1127.py")
ha = mm["H6a"]
rm = [ha[f"s{sd}"]["recall_m"] for sd in (0, 1, 2)]
dm = [ha[f"s{sd}"]["diff"] for sd in (0, 1, 2)]
em = [mm["H6b"]["sets"][f"m_s{sd}"]["e_15Hz"] for sd in (0, 1, 2)]
ec = [mm["H6b"]["sets"][f"crop_s{sd}"]["e_15Hz"] for sd in (0, 1, 2)]
add("Results 2; Limitations", "YOLO26m matched 36-37% of the reference boxes (at most 37%), 3-9 points more than the crop "
    "YOLO26s of the same seeds; 15 Hz e +1.40..+1.52 against +0.91..+1.32", "36-37%; 3-9; +1.40..+1.52; +0.91..+1.32",
    f"{rm}; {dm}; {em}; {ec}",
    (pct(min(rm)), pct(max(rm))) == (36, 37) and (hu(100 * min(dm), 0), hu(100 * max(dm), 0)) == (3, 9)
    and (r2(min(em)), r2(max(em))) == (1.40, 1.52) and (r2(min(ec)), r2(max(ec))) == (0.91, 1.32),
    "pa_followup_readout_1127.json: M/H6a, M/H6b/sets", "pa_followup_readout_1127.py")
rho9 = mdd["rho_e_nine"]
add("Results 2", "Nine sets: rho between matched fraction and e 0.83-0.93 over the four intervals (H6b at 15 Hz 0.833 >= 0.700); "
    "misses supplied alone raise |e| in all 12 YOLO26m seed x interval combinations (H6c)", "0.83-0.93; 12",
    f"{rho9}; H6b {mm['H6b']['rho']} ({mm['H6b']['verdict']}); H6c {mm['H6c']['worse']}/{mm['H6c']['n']} ({mm['H6c']['verdict']})",
    (r2(min(rho9.values())), r2(max(rho9.values()))) == (0.83, 0.93) and rho9["1"] == mm["H6b"]["rho"] >= 0.700
    and (mm["H6c"]["worse"], mm["H6c"]["n"]) == (12, 12), "pa_followup_readout_1127.json: M/descriptive/rho_e_nine, M/H6b, M/H6c",
    "pa_followup_readout_1127.py")
pp = mdd["paired_15Hz_3072"]
add("Results 2", "Per vine (21 pairs, 15 Hz, sigma 3072): YOLO26m matched more boxes in 15, higher e in 13, median e difference "
    "+0.23; median |difference| 0.32 against 0.44 between two crop YOLO26s seeds on the same vine", "15; 13; +0.23; 0.32; 0.44",
    f"{pp['higher_recall']}; {pp['higher_e']} (lower {pp['lower_e']}, equal {pp['equal_e']}); {mm['paired_15Hz_3072']['median_diff_e']}; "
    f"{pp['median_abs_diff_e']}; {mdd['crop_s_seed_pairs_15Hz_3072']['median_abs_diff_e']}",
    (pp["higher_recall"], pp["higher_e"], mm["paired_15Hz_3072"]["n"]) == (15, 13, 21)
    and r2(mm["paired_15Hz_3072"]["median_diff_e"]) == 0.23 and r2(pp["median_abs_diff_e"]) == 0.32
    and r2(mdd["crop_s_seed_pairs_15Hz_3072"]["median_abs_diff_e"]) == 0.44,
    "pa_followup_readout_1127.json: M/paired_15Hz_3072, M/descriptive", "pa_followup_readout_1127.py")
h6d = mm["H6d"]
add("Results 4", "YOLO26m: ratio on the other six vines left the median vine 28% off, one in ten 150% or more; does not beat the "
    "no-video baseline 23% / 46% (H6d)", "28%; 150%",
    f"{h6d['median']}; {h6d['p90']} (n {h6d['n']}); baseline {r2(h6d['baseline_median'])}/{r2(h6d['baseline_p90'])}; {h6d['verdict']}",
    (pct(h6d["median"]), pct(h6d["p90"]), h6d["n"]) == (28, 150, 420) and h6d["median"] >= h6d["baseline_median"]
    and h6d["p90"] >= h6d["baseline_p90"] and h6d["verdict"] == "unchanged with a larger detector",
    "pa_followup_readout_1127.json: M/H6d", "pa_followup_readout_1127.py")
pooled9 = pooled + rm
add("Abstract", "Nine detector sets (six YOLO26s, three YOLO26m) matched 22-37% of the reference boxes in a frame; misses supplied "
    "alone raised |e| for all nine (24/24 and 12/12 set x interval); at 15 Hz the sets that found more over-counted more "
    "(rho over nine 0.83)", "22-37%; nine",
    f"{min(pooled9)}-{max(pooled9)}; {worse}/24 + {mm['H6c']['worse']}/12; rho {mm['H6b']['rho']}",
    (pct(min(pooled9)), pct(max(pooled9))) == (22, 37) and worse == 24 and mm["H6c"]["worse"] == 12 and mm["H6b"]["rho"] >= 0.700,
    "supplementary_checks_1121.json: recall_s3072; pa_followup_readout_1127.json: M", "pa_followup_readout_1127.py")

out_csv = RES / "claims_pa_1121.csv"
with out_csv.open("w", newline="") as fh:
    wtr = csv.DictWriter(fh, fieldnames=list(rows[0]))
    wtr.writeheader()
    wtr.writerows(rows)
md = ["| # | Where | Claim | Printed | Frozen value | Pass | Source file: key | Tool |", "|---|---|---|---|---|---|---|---|"]
for i, r in enumerate(rows, 1):
    md.append(f"| {i} | {r['where']} | {r['claim']} | {r['printed']} | {r['computed']} | {r['ok']} | {r['source']} | {r['tool']} |")
(RES / "claims_pa_1121.md").write_text("\n".join(md) + "\n")
bad = [r for r in rows if r["ok"] != "yes"]
print(f"{len(rows)} claims, {len(bad)} failing")
for r in bad:
    print("FAIL:", r["claim"], "| printed", r["printed"], "| frozen", r["computed"])
sys.exit(1 if bad else 0)
