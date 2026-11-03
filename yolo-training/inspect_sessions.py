#!/usr/bin/env python3
"""核对标注端导出的 session，与 classes.yaml 的类别顺序是否一致。

标注是在另一台设备上做的，类别名/顺序很可能与预设不同。这个脚本只读不写，
逐个 session 报告：
  - 有没有 data.yaml / classes.txt，里面写的类别是什么；
  - 与 classes.yaml 相比是「完全一致 / 集合相同但顺序不同 / 类别不同」；
  - 实际标签里每个 class_id 出现多少次（能看出哪些类根本没标到）；
  - 图片尺寸分布与标签格式错误。

用法：
    py -3.14 yolo-training/inspect_sessions.py --sessions D:\\captures\\sessions

退出码：0 全部一致；1 没有类别声明；2 顺序不同；3 类别不同。
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import build_dataset as bd  # noqa: E402

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


def compare(seen: list[str], expected: list[str]) -> tuple[str, int, list[str]]:
    if seen == expected:
        return "ok", 0, []
    if set(seen) == set(expected):
        return "order", 2, []
    missing = [n for n in expected if n not in seen]
    extra = [n for n in seen if n not in expected]
    lines = []
    for tag, items in (("缺少", missing), ("多出", extra)):
        if not items:
            continue
        head = ", ".join(items[:8])
        tail = f" …（共 {len(items)} 类）" if len(items) > 8 else ""
        lines.append(f"{tag} {len(items)} 类：{head}{tail}")
    return "diff", 3, lines


def main() -> int:
    ap = argparse.ArgumentParser(description="核对 session 导出的类别与标签")
    ap.add_argument("--sessions", nargs="+", type=Path, required=True)
    ap.add_argument("--classes", type=Path, default=HERE / "classes.yaml")
    args = ap.parse_args()

    expected = bd.load_classes(args.classes)
    print(f"预设 {len(expected)} 类：{', '.join(expected)}")

    exports, warnings = bd.find_exports(args.sessions, Path("/__nonexistent__"))
    for w in warnings:
        print(f"[提示] {w}")
    if not exports:
        print("[没有找到] 目标目录下没有 images/ + labels/ 结构")
        return 2

    worst = 0
    for root in exports:
        print("=" * 72)
        print(f"session: {root}")
        splits = bd.splits_of(root)
        print(f"  split: {', '.join(splits) if splits else '无'}")

        seen, source = bd.read_export_names(root, expected)
        if seen is None:
            print("  类别声明: 无 data.yaml / classes.txt（无法核对名称，只能看 id）")
            worst = max(worst, 1)
        else:
            verdict, level, details = compare(seen, expected)
            worst = max(worst, level)
            label = {"ok": "与 classes.yaml 完全一致", "order": "类别相同但顺序不同", "diff": "类别不同"}[verdict]
            print(f"  类别声明: {source}")
            print(f"  -> {label}")
            for line in details:
                print(f"     {line}")
            if verdict == "order":
                print("     id 对照（标注端 -> 预设）：")
                for old_id, name in enumerate(seen):
                    print(f"       {old_id:>2} {name:<22} -> {expected.index(name):>2}")

        counts: Counter[int] = Counter()
        errors: list[str] = []
        sizes: Counter[str] = Counter()
        total = 0
        unlabeled = 0
        for split, (idir, ldir) in splits.items():
            if not idir.is_dir():
                continue
            for img in sorted(p for p in idir.iterdir() if p.is_file() and p.suffix.lower() in bd.IMG_EXTS):
                total += 1
                lbl = ldir / (img.stem + ".txt")
                if not lbl.is_file():
                    unlabeled += 1
                    continue
                records, errs = bd.parse_label_file(lbl, len(expected))
                errors.extend(f"{split}/{e}" for e in errs)
                counts.update(r[0] for r in records)
                if Image is not None:
                    try:
                        with Image.open(img) as im:
                            sizes[f"{im.width}x{im.height}"] += 1
                    except Exception:  # noqa: BLE001
                        sizes["unreadable"] += 1

        print(f"  图片 {total} 张，有标签 {total - unlabeled}，无标签 {unlabeled}；标签错误 {len(errors)} 条")
        if sizes:
            print(f"  尺寸分布: {dict(sizes.most_common(4))}")
        if errors:
            for e in errors[:5]:
                print(f"    [错误] {e}")
            if len(errors) > 5:
                print(f"    ... 另有 {len(errors) - 5} 条")

        used = {i: c for i, c in counts.items() if i < len(expected) and c > 0}
        if used:
            shown = ", ".join(f"{expected[i]}={c}" for i, c in sorted(used.items()))
            print(f"  有标签的类别 {len(used)}/{len(expected)}：{shown}")
            unused = [expected[i] for i in range(len(expected)) if counts.get(i, 0) == 0]
            if unused:
                head = ", ".join(unused[:8])
                tail = f" …（共 {len(unused)} 类没标到）" if len(unused) > 8 else ""
                print(f"  未出现: {head}{tail}")
        else:
            print("  [警告] 没有任何有效标签")

    print("=" * 72)
    if worst == 0:
        print("[结论] 类别声明与预设一致，可以直接 build_dataset.py")
    elif worst == 1:
        print("[结论] 无法核对类别名称，建议先确认标注端用的是哪套类别")
    elif worst == 2:
        print("[结论] 类别相同但顺序不同：直接 build 会让 id 错位，需要先统一顺序或做 id 重映射")
    else:
        print("[结论] 类别清单不同：先把两套类别对齐（改名/合并/删除），再构建数据集")
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
