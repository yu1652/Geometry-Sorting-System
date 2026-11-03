#!/usr/bin/env python3
"""在服务器上造一个极小的合成数据集，用来验证「训练链路 + GPU」是否通。

只依赖 Pillow，不碰真实数据。生成的 data.yaml 与 build_dataset.py 的格式一致
（不写 path / nc），类别取自 classes.yaml，因此顺带验证 26 类名在 Linux 上可用。

用法：
    /root/miniconda3/bin/python make_smoke_dataset.py --out /root/autodl-tmp/ylers/datasets/smoke
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

COLORS = [
    (200, 60, 60), (60, 170, 60), (60, 90, 200), (200, 150, 40),
    (150, 60, 180), (40, 160, 160), (120, 120, 120), (210, 100, 140),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--classes", type=Path, default=Path(__file__).with_name("classes.yaml"))
    ap.add_argument("--train", type=int, default=8)
    ap.add_argument("--val", type=int, default=2)
    ap.add_argument("--width", type=int, default=160)
    ap.add_argument("--height", type=int, default=96)
    args = ap.parse_args()

    names = yaml.safe_load(args.classes.read_text(encoding="utf-8"))["names"]
    rng = random.Random(0)
    out = args.out

    def make(split: str, count: int) -> None:
        idir = out / "images" / split
        ldir = out / "labels" / split
        idir.mkdir(parents=True, exist_ok=True)
        ldir.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            img = Image.new("RGB", (args.width, args.height), (235, 235, 235))
            draw = ImageDraw.Draw(img)
            lines = []
            for _ in range(rng.randint(1, 2)):
                cls = rng.randrange(len(names))
                bw = rng.uniform(0.18, 0.35)
                bh = rng.uniform(0.25, 0.55)
                cx = rng.uniform(bw / 2 + 0.05, 1 - bw / 2 - 0.05)
                cy = rng.uniform(bh / 2 + 0.05, 1 - bh / 2 - 0.05)
                x1, y1 = (cx - bw / 2) * args.width, (cy - bh / 2) * args.height
                x2, y2 = (cx + bw / 2) * args.width, (cy + bh / 2) * args.height
                draw.rectangle([x1, y1, x2, y2], fill=COLORS[cls % len(COLORS)], outline=(25, 25, 25), width=2)
                lines.append(f"{cls} {cx:.4f} {cy:.4f} {bw:.4f} {bh:.4f}\n")
            stem = f"{split}_{i:03d}"
            img.save(idir / f"{stem}.jpg", quality=90)
            (ldir / f"{stem}.txt").write_text("".join(lines), encoding="utf-8")

    make("train", args.train)
    make("val", args.val)

    text = ["# 合成冒烟数据集，验证完即可删除。", "", "train: images/train", "val: images/val", "", "names:"]
    text += [f"  - {n}" for n in names]
    (out / "data.yaml").write_text("\n".join(text) + "\n", encoding="utf-8")
    print(f"已生成 {args.train} train / {args.val} val 张，类别 {len(names)} 个 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
