"""Tiled and whole-frame reading of the same footage (GrapeMOTS held-out pair PP2+PP4,
one checkpoint), each scored under the intervention's design: the dense arm (every
labelled frame) is scored only at the frames the sparse arm (every kth) processed.
Input: grapemots-protocol cadence2026_0919/raw/heldout.tar.gz."""
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from decompose_count_error import decompose
RAW = Path(sys.argv[2])
PER = ("frame_predicted_ids", "frame_predicted_boxes", "frame_gt_ids", "frame_gt_boxes")
FILES = {"8 tiles": {1: "tiled_s1", 4: "tiled_s4", 8: "tiled_s8"},
         "whole frame 3840": {1: "grid_3840_s1", 4: "grid_3840_s4", 8: "grid_3840_s8"},
         "whole frame 3072": {1: "grid_3072_s1", 4: "pred_full3072_s4", 8: "grid_3072_s8"}}
def load(name):
    d = json.loads((RAW / f"{name}.json").read_text())
    return d["config"].get("weights"), {v["video"]: v for v in d["videos"]}
out = {}
for mode, files in FILES.items():
    w1, dense = load(files[1])
    for k in (4, 8):
        wk, sparse = load(files[k]); assert wk == w1, (mode, k)
        rows = []
        for vid, v in dense.items():
            sub = dict(v); [sub.__setitem__(key, v[key][0::k]) for key in PER]
            assert sub["frame_gt_ids"] == sparse[vid]["frame_gt_ids"]
            a, b = decompose(sub, 0.5, 1), decompose(sparse[vid], 0.5, 1)
            rows.append((vid, a, b))
        G = sum(a["G"] for _, a, _ in rows)
        Pd = sum(a["P"] for _, a, _ in rows); Ps = sum(b["P"] for _, _, b in rows)
        out[f"{mode}, k={k}"] = {"weights": w1, "G": G, "P_dense": Pd, "P_sparse": Ps,
            "e_dense": round((Pd - G) / G, 3), "e_sparse": round((Ps - G) / G, 3),
            "per_sequence": {vid: [a["P"], b["P"], a["G"]] for vid, a, b in rows}}
Path(sys.argv[3]).write_text(json.dumps(out, indent=1))
for k, v in out.items():
    print(f"{k:24} G={v['G']:3d}  P dense/sparse {v['P_dense']:3d}/{v['P_sparse']:3d}   e {v['e_dense']:+.3f}/{v['e_sparse']:+.3f}   per seq {v['per_sequence']}")
print("same checkpoint across modes:", len({v['weights'] for v in out.values()}) == 1)
