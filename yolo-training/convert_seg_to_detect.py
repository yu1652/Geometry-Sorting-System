#!/usr/bin/env python3
"""把 YOLO 实例分割标签（多边形）转成 YOLO 检测标签（bbox）。

背景：标注端导出的是实例分割 txt（`class_id x1 y1 x2 y2 ...`），
而部署端用的是检测模型（输出 1×(4+nc)×8400 的框）。多边形取外接矩形即可，
类别 id 不变，因此可以直接用现有 classes.yaml 训练检测模型。

转换规则：
  - 每行取所有点的 min/max，得到归一化 bbox：cx cy w h；
  - 宽或高为 0 的退化成框会被丢弃并计数（通常是标错或点重合）；
  - 坐标超出 [0,1] 的会被截断到边界并计数；
  - 图片默认复制，`--symlink` 可改成软链接；文件名保持不变。

用法：
    py -3.14 yolo-training/convert_seg_to_detect.py --src <原始目录> --out <输出目录>
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NAME_RE = None


def convert_line(line: str) -> tuple[tuple[int, float, float, float, float] | None, str | None, bool]:
    """返回 (bbox 或 None, 错误信息, 是否发生截断)。"""
    parts = line.split()
    if len(parts) < 7:  # class + 至少 3 个点才可能是多边形
        if len(parts) == 5:
            # 已经是检测框，原样返回
            try:
                cls = int(float(parts[0]))
                cx, cy, w, h = (float(x) for x in parts[1:])
            except ValueError:
                return None, "数值解析失败", False
            return (cls, cx, cy, w, h), None, False
        return None, f"列数不足（{len(parts)}）", False
    try:
        cls = int(float(parts[0]))
        coords = [float(x) for x in parts[1:]]
    except ValueError:
        return None, "数值解析失败", False
    if len(coords) % 2:
        return None, "坐标个数不是偶数", False
    xs, ys = coords[0::2], coords[1::2]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    clipped = False
    for v in (x1, y1, x2, y2):
        if v < 0 or v > 1:
            clipped = True
    x1, x2 = max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))
    y1, y2 = max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))
    w, h = x2 - x1, y2 - y1
    if w <= 1e-6 or h <= 1e-6:
        return None, "外接框宽或高为 0", clipped
    return ((cls, (x1 + x2) / 2, (y1 + y2) / 2, w, h), None, clipped)


def main() -> int:
    ap = argparse.ArgumentParser(description="分割多边形标签 -> 检测框标签")
    ap.add_argument("--src", type=Path, required=True, help="含 images/ 和 labels/ 的目录（也兼容 images/train 等）")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--classes", type=Path, default=Path(__file__).with_name("classes.yaml"))
    ap.add_argument("--symlink", action="store_true", help="用软链接代替复制图片")
    ap.add_argument("--force", action="store_true", help="输出目录非空时清空重来")
    args = ap.parse_args()

    names = yaml.safe_load(args.classes.read_text(encoding="utf-8"))["names"]
    nc = len(names)

    if args.out.exists() and any(args.out.iterdir()):
        if not args.force:
            print(f"[错误] 输出目录非空：{args.out}（加 --force 清空）", file=sys.stderr)
            return 4
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    per_class = Counter()
    errors: list[str] = []
    src_images = [p for p in args.src.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    seen_pairs = 0

    for img in src_images:
        rel = img.relative_to(args.src)
        # 对应标签：把路径里的 images 换成 labels，扩展名换成 .txt
        parts = list(rel.parts)
        if "images" in parts:
            parts[parts.index("images")] = "labels"
        rel_lbl = Path(*parts).with_suffix(".txt")
        lbl = args.src / rel_lbl
        if not lbl.is_file():
            alt = args.src / "labels" / rel.name
            lbl = alt if alt.is_file() else lbl
        if not lbl.is_file():
            errors.append(f"{img.name}: 找不到标签")
            stats["无标签"] += 1
            continue

        rows = []
        for lineno, raw in enumerate(lbl.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if not raw.strip():
                continue
            box, err, clipped = convert_line(raw)
            if err:
                errors.append(f"{lbl.name}:{lineno} {err}")
                stats["坏行"] += 1
                continue
            if clipped:
                stats["截断"] += 1
            cls = box[0]
            if not 0 <= cls < nc:
                errors.append(f"{lbl.name}:{lineno} class_id={cls} 超出 0..{nc - 1}")
                stats["越界类"] += 1
                continue
            rows.append(box)
            per_class[cls] += 1
            stats["框"] += 1

        if not rows:
            stats["空标签"] += 1

        out_lbl = args.out / rel_lbl
        out_lbl.parent.mkdir(parents=True, exist_ok=True)
        out_lbl.write_text(
            "".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for c, cx, cy, w, h in rows),
            encoding="utf-8",
        )
        out_img = args.out / rel
        out_img.parent.mkdir(parents=True, exist_ok=True)
        if args.symlink:
            if out_img.exists():
                out_img.unlink()
            out_img.symlink_to(img.resolve())
        else:
            shutil.copy2(img, out_img)
        seen_pairs += 1
        stats["图片"] += 1

    print(f"来源 {args.src}")
    print(f"输出 {args.out}")
    print(f"处理图片 {stats['图片']} 张，转换出 {stats['框']} 个框；无标签 {stats['无标签']}，空标签 {stats['空标签']}")
    print(f"坏行 {stats['坏行']}，class 越界 {stats['越界类']}，坐标截断 {stats['截断']}")
    used = [(i, per_class[i]) for i in range(nc) if per_class.get(i)]
    print(f"有框的类别 {len(used)}/{nc}：" + ", ".join(f"{names[i]}={c}" for i, c in used))
    missing = [names[i] for i in range(nc) if not per_class.get(i)]
    if missing:
        print(f"没有框的类别：{', '.join(missing)}")
    if errors:
        print(f"\n前 10 条问题：")
        for e in errors[:10]:
            print("  " + e)
    print("\n下一步（构建训练集）：")
    print(f"  py -3.14 yolo-training/build_dataset.py --sessions \"{args.out}\" --out <数据集根>")
    return 0 if stats["框"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
