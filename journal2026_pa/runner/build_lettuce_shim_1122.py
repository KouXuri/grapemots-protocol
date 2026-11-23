#!/usr/bin/env python3
"""Present LettuceMOTS in the layout create_grapemots_detection_dataset.py expects.

LettuceMOTS (Hu et al., Journal of Field Robotics 2024; github.com/NanH5837/LettuceMOTS) ships as
<root>/{train,test}/{images,instances}/<seq>/NNNNNN.png with uint16 KITTI-MOTS masks
(value = class*1000 + track, class 1 = lettuce).  GrapeMOTS ships as
<root>/<Video>/default/{images,instances}/...  Same encoding, different nesting, so a tree of
symlinks is enough and no pixels are copied (as tools/build_apple_shim.py does for APPLE MOTS).

The data are used only for the association-alone (oracle) check, which needs no training, so
every sequence goes to one split ("test") and no detector is trained on them.

    .venv/bin/python tools/build_lettuce_shim_1122.py
    .venv/bin/python tools/create_grapemots_detection_dataset.py --src /home/kou/my_env/LettuceMOTS_shim \
        --dst datasets/lettucemots_det --classes lettuce --split-json datasets/lettucemots_split_1122.json
"""
from pathlib import Path
import json

SRC = Path("/home/kou/my_env/lettucemots")
SHIM = Path("/home/kou/my_env/LettuceMOTS_shim")

seqs = {}
for split in ("train", "test"):
    for d in sorted((SRC / split / "images").iterdir()):
        if d.is_dir():
            seqs[d.name] = split

SHIM.mkdir(exist_ok=True)
for seq, split in sorted(seqs.items()):
    base = SHIM / seq / "default"
    base.mkdir(parents=True, exist_ok=True)
    img_link = base / "images"
    if not img_link.exists():
        img_link.symlink_to(SRC / split / "images" / seq)
    inst = base / "instances"
    inst.mkdir(exist_ok=True)
    (inst / "labels.txt").write_text("lettuce\n")
    n = 0
    for m in sorted((SRC / split / "instances" / seq).glob("*.png")):
        link = inst / m.name
        if not link.exists():
            link.symlink_to(m)
        n += 1
    print(f"{seq} ({split}): {n} masks, {len(list((SRC / split / 'images' / seq).glob('*.png')))} images")

out = Path("/home/kou/my_env/yolo26/datasets/lettucemots_split_1122.json")
out.write_text(json.dumps({"test": sorted(seqs)}, indent=1) + "\n")
print("wrote", out)
