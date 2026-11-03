#!/usr/bin/env python3
"""把双相机采集器导出的多个 session 合成一个可直接训练的 YOLO 数据集。

它会做四件事：
  1. 扫描 session 导出目录（形如 .../session_*/yolo_dataset/images + labels）；
  2. 校验标签（列数、class_id 范围、坐标范围），发现类别顺序与 classes.yaml 不一致时拒绝；
  3. 按主类分层切 train/val（保证稀有类不会被切空）；
  4. 生成 <数据集根>/data.yaml —— 故意不写 path 和 nc：
       - 没有 path 时，Ultralytics 以 data.yaml 所在目录作为数据集根，整目录可搬到 Linux；
       - 没有 nc 时，nc 自动等于 len(names)，类别表改动后不会出现 nc/names 不一致。

三种模式：
  --init   在还没采集/还没标注时，先建好目录骨架和占位 data.yaml
  --check  只校验一个已经建好的数据集，不写任何文件
  （默认） 从 --sessions 扫描并构建到 --out

退出码：0 正常；2 没找到可用数据；3 校验不通过；4 参数/IO 问题。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("缺少 PyYAML，请先执行：pip install pyyaml")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_NAMES = {"train", "val", "valid", "test"}
SKIP_DIRS = {"_internal", ".git", "node_modules", "__pycache__", "runs", "weights"}
NAME_RE = re.compile(r"^[a-z0-9_]+$")
COORD_EPS = 1e-6


# --------------------------------------------------------------------------- #
# 基础类型
# --------------------------------------------------------------------------- #
@dataclass
class Item:
    """一张图 + 它的标签。"""

    export: str
    name: str                      # 输出用的文件名（不含扩展名）
    src_img: Path
    src_lbl: Path | None
    records: list[tuple[int, float, float, float, float]] = field(default_factory=list)

    @property
    def primary_cls(self) -> int:
        """主类：第一行标签的 class_id；无标签记 -1（背景负样本）。"""
        return self.records[0][0] if self.records else -1


@dataclass
class Export:
    root: Path
    splits: dict[str, tuple[Path, Path]]  # split -> (images_dir, labels_dir)
    names: list[str] | None = None
    names_source: Path | None = None


# --------------------------------------------------------------------------- #
# 配置读取
# --------------------------------------------------------------------------- #
def load_classes(path: Path) -> list[str]:
    if not path.is_file():
        raise SystemExit(f"[错误] 找不到类别表：{path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    names = data.get("names")
    if not isinstance(names, list) or not names:
        raise SystemExit(f"[错误] {path} 里需要一个非空的 names 列表")
    names = [str(n) for n in names]
    if len(set(names)) != len(names):
        dup = [n for n, c in Counter(names).items() if c > 1]
        raise SystemExit(f"[错误] 类别名重复：{dup}（YOLO 的 class_id 必须一一对应）")
    bad = [n for n in names if not NAME_RE.match(n)]
    if bad:
        raise SystemExit(
            f"[错误] 类别名只能用英文小写 + 数字 + 下划线，以下不合法：{bad}\n"
            f"       中文名会触发字体下载，且与部署端 id 映射对不上。"
        )
    return names


def normalize_names(raw) -> list[str] | None:
    """把 {'0': 'cube', ...} 或 ['cube', ...] 统一成有序列表。"""
    if isinstance(raw, dict):
        try:
            table = {int(k): str(v) for k, v in raw.items()}
        except (TypeError, ValueError):
            return None
        keys = sorted(table)
        if keys != list(range(len(keys))):
            return None
        return [table[k] for k in keys]
    if isinstance(raw, list):
        return [str(n) for n in raw]
    return None


def read_export_names(export_dir: Path, root_names: list[str]) -> tuple[list[str] | None, Path | None]:
    """从 session 自带 data.yaml / classes.txt 读类别顺序，用于防漂移比对。"""
    candidates = [
        export_dir / "data.yaml",
        export_dir / "dataset.yaml",
        export_dir / "classes.txt",
        export_dir / "labels" / "classes.txt",
    ]
    for cand in candidates:
        if not cand.is_file():
            continue
        try:
            if cand.suffix.lower() == ".txt":
                names = [ln.strip() for ln in cand.read_text(encoding="utf-8").splitlines() if ln.strip()]
            else:
                data = yaml.safe_load(cand.read_text(encoding="utf-8")) or {}
                names = normalize_names(data.get("names"))
                if names is None and isinstance(data.get("nc"), int):
                    # 只有 nc 没有 names 时无法比对顺序，跳过
                    continue
            if names:
                return names, cand
        except Exception:  # noqa: BLE001 - 坏 yaml 不该让整个流程崩
            continue
    return None, None


# --------------------------------------------------------------------------- #
# 扫描
# --------------------------------------------------------------------------- #
def find_exports(roots: list[Path], out_dir: Path) -> tuple[list[Path], list[str]]:
    """找出所有同时含 images/ 和 labels/ 的导出目录。"""
    found: list[Path] = []
    warnings: list[str] = []
    out_resolved = out_dir.resolve()
    for root in roots:
        root = root.expanduser()
        if not root.is_dir():
            warnings.append(f"跳过不存在的路径：{root}")
            continue
        for dirpath, dirnames, _ in os.walk(root):
            here = Path(dirpath)
            try:
                here_resolved = here.resolve()
            except OSError:
                dirnames[:] = []
                continue
            if here_resolved == out_resolved or out_resolved in here_resolved.parents:
                dirnames[:] = []          # 别把输出目录当输入
                continue
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
            if "images" in dirnames and "labels" in dirnames:
                found.append(here)
                dirnames[:] = []          # 找到导出目录后不再往里走
    uniq = sorted(set(found))
    if not uniq:
        warnings.append(
            "没有找到同时包含 images/ 和 labels/ 的导出目录。\n"
            "  期望形式：<...>/session_20260528_220342/yolo_dataset/images + labels"
        )
    return uniq, warnings


def splits_of(export_dir: Path) -> dict[str, tuple[Path, Path]]:
    """拆出 split -> (images, labels)，兼容 images/train 与平铺 images。"""
    images_root = export_dir / "images"
    labels_root = export_dir / "labels"
    splits: dict[str, tuple[Path, Path]] = {}

    if not images_root.is_dir():
        return splits

    subdirs = [d for d in sorted(images_root.iterdir()) if d.is_dir()]
    named = [d for d in subdirs if d.name.lower() in SPLIT_NAMES]
    if named:
        for d in named:
            split = "val" if d.name.lower() in {"val", "valid"} else d.name.lower()
            lbl = labels_root / d.name
            if not lbl.is_dir():
                lbl = labels_root
            splits[split] = (d, lbl)
    elif subdirs:
        # 不常见的布局：images/<任意名字>/；按子目录名当 split
        for d in subdirs:
            lbl = labels_root / d.name
            splits[d.name] = (d, lbl if lbl.is_dir() else labels_root)
    else:
        splits["all"] = (images_root, labels_root)
    return splits


def parse_label_file(path: Path, nc: int) -> tuple[list[tuple[int, float, float, float, float]], list[str]]:
    records: list[tuple[int, float, float, float, float]] = []
    errors: list[str] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return [], [f"读不了标签文件 {path.name}: {exc}"]

    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{path.name}:{lineno} 列数={len(parts)}，应为 5（cls cx cy w h）")
            continue
        try:
            cls = int(float(parts[0]))
        except ValueError:
            errors.append(f"{path.name}:{lineno} class_id 不是整数：{parts[0]!r}")
            continue
        if not 0 <= cls < nc:
            errors.append(f"{path.name}:{lineno} class_id={cls} 超出范围 0..{nc - 1}")
            continue
        try:
            cx, cy, w, h = (float(x) for x in parts[1:])
        except ValueError:
            errors.append(f"{path.name}:{lineno} 坐标不是数字：{parts[1:]!r}")
            continue
        if w <= 0 or h <= 0:
            errors.append(f"{path.name}:{lineno} 宽/高必须大于 0（w={w}, h={h}）")
            continue
        if not all(-COORD_EPS <= v <= 1 + COORD_EPS for v in (cx, cy, w, h)):
            errors.append(
                f"{path.name}:{lineno} 坐标应在 0~1 归一化范围内（{cx},{cy},{w},{h}）；"
                f"像素坐标需要先归一化"
            )
            continue
        records.append((cls, cx, cy, w, h))
    return records, errors


# --------------------------------------------------------------------------- #
# 收集
# --------------------------------------------------------------------------- #
def collect(
    exports: list[Export], nc: int, keep_empty: bool, dedupe: bool
) -> tuple[list[Item], list[str], list[str], Counter[str]]:
    items: list[Item] = []
    errors: list[str] = []
    warnings: list[str] = []
    used_names: set[str] = set()
    kept_empty = 0
    dropped_empty = 0

    for export in exports:
        for split, (idir, ldir) in export.splits.items():
            images = sorted(p for p in idir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)
            if not images:
                continue
            if not ldir.is_dir():
                warnings.append(f"{export.root.name}/{split}: 没有 labels 目录，跳过 {len(images)} 张图")
                continue
            for img in images:
                lbl = ldir / (img.stem + ".txt")
                records: list[tuple[int, float, float, float, float]] = []
                if lbl.is_file():
                    records, errs = parse_label_file(lbl, nc)
                    errors.extend(f"{export.root.name}/{split}/{e}" for e in errs)
                if not records:
                    if keep_empty:
                        kept_empty += 1
                    else:
                        dropped_empty += 1
                        continue

                base = img.stem
                name = base
                if name in used_names:
                    name = f"{export.root.name}__{base}"
                    i = 2
                    while name in used_names:
                        name = f"{export.root.name}__{i}__{base}"
                        i += 1
                used_names.add(name)
                items.append(
                    Item(
                        export=export.root.name,
                        name=name,
                        src_img=img,
                        src_lbl=lbl if records and lbl.is_file() else None,
                        records=records,
                    )
                )

    if kept_empty:
        warnings.append(f"{kept_empty} 张图没有标签，已按背景负样本保留（--drop-empty 可丢弃）")
    if dropped_empty:
        warnings.append(f"{dropped_empty} 张图没有标签，已丢弃（--keep-empty 可保留）")

    if dedupe and items:
        seen: dict[str, Item] = {}
        dup = 0
        for it in items:
            digest = hashlib.sha256(it.src_img.read_bytes()).hexdigest()
            if digest in seen:
                dup += 1
                continue
            seen[digest] = it
        items = list(seen.values())
        if dup:
            warnings.append(f"--dedupe 去掉了 {dup} 张内容完全相同的图片")

    per_export = Counter(it.export for it in items)
    return items, errors, warnings, per_export


# --------------------------------------------------------------------------- #
# 切分
# --------------------------------------------------------------------------- #
def split_items(items: list[Item], val_ratio: float) -> tuple[list[Item], list[Item]]:
    """按主类分层：每组里每 stride 张取一张进 val。

    稀有类（组内不足 stride 张）全部留在 train，避免验证集里只有 1 张、指标失去意义。
    小数据集（总数不足 stride 张/类）分层会让 val 过少，这里再按比例补齐：
    从训练样本最多的类里挪，且保证每个类至少留 1 张在 train。
    """
    stride = max(2, round(1.0 / val_ratio))
    groups: dict[int, list[Item]] = defaultdict(list)
    for it in items:
        groups[it.primary_cls].append(it)

    assign: dict[str, str] = {}
    for cls in sorted(groups):
        group = sorted(groups[cls], key=lambda x: x.name)
        if len(group) < stride:
            assign.update({it.name: "train" for it in group})
            continue
        for idx, it in enumerate(group):
            assign[it.name] = "val" if idx % stride == 0 else "train"

    # 按比例补齐 val
    target = max(1, round(len(items) * val_ratio))
    n_val = sum(1 for v in assign.values() if v == "val")
    if n_val < target:
        per_class_train = Counter(it.primary_cls for it in items if assign[it.name] == "train")
        for it in sorted(items, key=lambda x: (-per_class_train[x.primary_cls], x.name)):
            if n_val >= target:
                break
            if assign[it.name] == "val" or per_class_train[it.primary_cls] <= 1:
                continue
            assign[it.name] = "val"
            per_class_train[it.primary_cls] -= 1
            n_val += 1

    # 极端情况兜底：至少各留一张
    if all(v == "val" for v in assign.values()):
        assign[sorted(assign)[0]] = "train"
    if not any(v == "val" for v in assign.values()):
        assign[sorted(assign)[-1]] = "val"

    train = sorted((it for it in items if assign[it.name] == "train"), key=lambda x: x.name)
    val = sorted((it for it in items if assign[it.name] == "val"), key=lambda x: x.name)
    return train, val


# --------------------------------------------------------------------------- #
# 写出
# --------------------------------------------------------------------------- #
def data_yaml_text(names: list[str], has_test: bool = False) -> str:
    lines = [
        "# 由 yolo-training/build_dataset.py 生成，不要手改。",
        "# 改类别请改 classes.yaml 后重跑；本文件故意不写 path / nc：",
        "#   数据集根 = 本文件所在目录；nc 自动等于 len(names)。",
        "",
        "train: images/train",
        "val: images/val",
    ]
    if has_test:
        lines.append("test: images/test")
    lines += ["", "names:"]
    lines += [f"  - {n}" for n in names]
    return "\n".join(lines) + "\n"


def write_dataset(
    out: Path,
    names: list[str],
    train: list[Item],
    val: list[Item],
    mode: str,
    force: bool,
) -> Counter[str]:
    stats: Counter[str] = Counter()
    if out.exists() and any(out.iterdir()):
        if not force:
            raise SystemExit(
                f"[错误] 输出目录非空：{out}\n"
                f"       换个 --out，或加 --force 只清理本工具生成的文件后重建。"
            )
        for sub in ("images", "labels"):
            if (out / sub).exists():
                shutil.rmtree(out / sub)
        for f in ("data.yaml", "dataset_report.json"):
            if (out / f).exists():
                (out / f).unlink()

    for split, rows in (("train", train), ("val", val)):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for it in rows:
            dst_img = out / "images" / split / f"{it.name}{it.src_img.suffix.lower()}"
            dst_lbl = out / "labels" / split / f"{it.name}.txt"
            if mode == "symlink":
                try:
                    if dst_img.exists():
                        dst_img.unlink()
                    os.symlink(it.src_img.resolve(), dst_img)
                    stats["symlink"] += 1
                except OSError:
                    shutil.copy2(it.src_img, dst_img)
                    stats["copy_fallback"] += 1
            else:
                shutil.copy2(it.src_img, dst_img)
                stats["copy"] += 1
            # 无标签图片写空 txt：YOLO 把它当背景负样本
            text = ""
            if it.records:
                text = "".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for c, cx, cy, w, h in it.records)
            dst_lbl.write_text(text, encoding="utf-8")
            stats[split] += 1

    (out / "data.yaml").write_text(data_yaml_text(names), encoding="utf-8")
    return stats


def image_sizes(items: list[Item], limit: int = 500) -> dict[str, int]:
    """统计图片尺寸分布 —— 用来确认导出的是双拼图还是单视图。"""
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover
        return {}
    sizes: Counter[str] = Counter()
    for it in items[:limit]:
        try:
            with Image.open(it.src_img) as im:
                sizes[f"{im.width}x{im.height}"] += 1
        except Exception:  # noqa: BLE001
            sizes["unreadable"] += 1
    return dict(sizes.most_common())


def build_report(
    names: list[str],
    train: list[Item],
    val: list[Item],
    warnings: list[str],
    errors: list[str],
    exports: list[Export],
    per_export: Counter[str],
) -> dict:
    def class_counts(rows: list[Item]) -> dict[str, int]:
        c: Counter[int] = Counter()
        for it in rows:
            c.update(r[0] for r in it.records)
        return {names[i]: c.get(i, 0) for i in range(len(names))}

    all_items = train + val
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "classes": names,
        "nc": len(names),
        "images": {"train": len(train), "val": len(val), "total": len(all_items)},
        "labeled_images": sum(1 for it in all_items if it.records),
        "background_images": sum(1 for it in all_items if not it.records),
        "per_class": {"train": class_counts(train), "val": class_counts(val)},
        "per_export": dict(sorted(per_export.items())),
        "image_sizes": image_sizes(all_items),
        "sources": [str(e.root) for e in exports],
        "warnings": warnings,
        "errors": errors[:500],
        "error_count": len(errors),
    }


# --------------------------------------------------------------------------- #
# 模式：init / check
# --------------------------------------------------------------------------- #
def do_init(out: Path, names: list[str], force: bool) -> int:
    if out.exists() and any(out.iterdir()) and not force:
        print(f"[错误] 输出目录非空：{out}（加 --force 只清理本工具的文件）", file=sys.stderr)
        return 4
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    (out / "data.yaml").write_text(data_yaml_text(names), encoding="utf-8")
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "classes": names,
        "nc": len(names),
        "images": {"train": 0, "val": 0, "total": 0},
        "note": "--init 只搭了空骨架，还没有图片；采集 + 标注后重跑 build_dataset.py",
    }
    (out / "dataset_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[完成] 已建好空数据集骨架：{out}")
    print(f"       类别 {len(names)} 个：{', '.join(names)}")
    print(f"       下一步：采集 → 标注 →")
    print(f"       py -3.14 yolo-training/build_dataset.py --sessions <会话根> --out \"{out}\"")
    return 0


def do_check(out: Path, names: list[str]) -> int:
    yaml_path = out / "data.yaml"
    if not yaml_path.is_file():
        print(f"[错误] 没有 data.yaml：{out}", file=sys.stderr)
        return 4
    try:
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        print(f"[错误] data.yaml 解析失败：{exc}", file=sys.stderr)
        return 3

    errors: list[str] = []
    warnings: list[str] = []
    if "path" in data:
        warnings.append("data.yaml 里写了 path，会让数据集根不可移植；建议删掉（本工具生成的不写 path）")
    if "nc" in data and data["nc"] != len(names):
        errors.append(f"data.yaml nc={data['nc']} 与类别表长度 {len(names)} 不一致")

    total = 0
    for split in ("train", "val", "test"):
        rel = data.get(split)
        if not rel:
            continue
        idir = (out / rel).resolve()
        parts = list(idir.parts)
        for i in range(len(parts) - 1, -1, -1):
            if parts[i] == "images":
                parts[i] = "labels"
                break
        ldir = Path(*parts)
        if not idir.is_dir():
            errors.append(f"{split}: 图片目录不存在 {idir}")
            continue
        images = sorted(p for p in idir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)
        if not images:
            warnings.append(f"{split}: 目录是空的")
        missing = 0
        for img in images:
            total += 1
            lbl = ldir / (img.stem + ".txt")
            if not lbl.is_file():
                missing += 1
                continue
            _, errs = parse_label_file(lbl, len(names))
            errors.extend(f"{split}/{e}" for e in errs)
        if missing:
            warnings.append(f"{split}: {missing} 张图没有标签文件（会被当背景负样本）")

    print(f"[检查] {out}")
    print(f"       类别 {len(names)} 个：{', '.join(names)}")
    print(f"       图片 {total} 张；错误 {len(errors)} 条；提示 {len(warnings)} 条")
    for w in warnings:
        print(f"       [提示] {w}")
    for e in errors[:30]:
        print(f"       [错误] {e}")
    if len(errors) > 30:
        print(f"       ... 另有 {len(errors) - 30} 条错误，见上")
    if errors:
        return 3
    print("       [通过] data.yaml 与标签校验正常")
    return 0


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="把多个 session 导出合成一个可直接训练的 YOLO 数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  py -3.14 build_dataset.py --init --out D:/datasets/sort/shape\n"
            "  py -3.14 build_dataset.py --sessions D:/captures/dataset/sessions --out D:/datasets/sort/shape\n"
            "  py -3.14 build_dataset.py --check --out D:/datasets/sort/shape\n"
        ),
    )
    parser.add_argument("--sessions", nargs="+", type=Path, help="采集会话根目录（可多个），递归找 images/+labels/")
    parser.add_argument("--out", type=Path, required=True, help="输出数据集根目录")
    parser.add_argument("--classes", type=Path, default=Path(__file__).with_name("classes.yaml"), help="类别表")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例（默认 0.2）")
    parser.add_argument("--seed", type=int, default=0, help="保留参数；切分本身是确定性的")
    parser.add_argument("--mode", choices=("copy", "symlink"), default="copy", help="图片写入方式（默认 copy）")
    parser.add_argument("--keep-empty", dest="keep_empty", action="store_true", default=True, help="保留无标签图为背景（默认）")
    parser.add_argument("--drop-empty", dest="keep_empty", action="store_false", help="丢弃无标签图")
    parser.add_argument("--dedupe", action="store_true", help="按内容哈希去重")
    parser.add_argument("--force", action="store_true", help="允许清理非空输出目录里本工具生成的文件")
    parser.add_argument("--force-names", action="store_true", help="忽略 session 自带类别表与 classes.yaml 的不一致")
    parser.add_argument("--dry-run", action="store_true", help="只统计，不写文件")
    parser.add_argument("--init", action="store_true", help="只建空骨架 + data.yaml")
    parser.add_argument("--check", action="store_true", help="只校验已有数据集")
    args = parser.parse_args(argv)

    if args.init and args.check:
        print("[错误] --init 与 --check 只能选一个", file=sys.stderr)
        return 4
    if not 0 < args.val_ratio < 0.5:
        print(f"[错误] --val-ratio 应在 (0, 0.5) 之间，当前 {args.val_ratio}", file=sys.stderr)
        return 4

    names = load_classes(args.classes)
    out: Path = args.out.expanduser()

    if args.init:
        return do_init(out, names, args.force)
    if args.check:
        return do_check(out, names)

    if not args.sessions:
        print("[错误] 需要 --sessions（或用 --init / --check）", file=sys.stderr)
        return 4

    exports_roots, warnings = find_exports([p for p in args.sessions], out)
    if not exports_roots:
        for w in warnings:
            print(f"[提示] {w}")
        print("\n[没有数据] 还没有可训练的数据集。")
        print("  1) 先用双相机采集器采集，再用 X-AnyLabeling 导出 YOLO txt；")
        print("  2) 或者先建空骨架看目录结构：")
        print(f"     py -3.14 yolo-training/build_dataset.py --init --out \"{out}\"")
        return 2

    exports: list[Export] = []
    conflicts: list[str] = []
    for root in exports_roots:
        splits = splits_of(root)
        if not splits:
            warnings.append(f"{root}: 没有可识别的 images 结构，跳过")
            continue
        seen, source = read_export_names(root, names)
        if seen and seen != names:
            conflicts.append(
                f"{root}\n    该 session 的类别顺序（{source}）：{seen}\n"
                f"    classes.yaml 的顺序：{names}"
            )
        exports.append(Export(root=root, splits=splits, names=seen, names_source=source))

    if conflicts and not args.force_names:
        print("[错误] 发现类别顺序不一致，继续下去会让 class_id 整体错位：", file=sys.stderr)
        for c in conflicts:
            print(f"  - {c}", file=sys.stderr)
        print("  确认要以 classes.yaml 为准时，加 --force-names。", file=sys.stderr)
        return 3
    if conflicts:
        warnings.extend(f"类别顺序不一致但已 --force-names 忽略：{c.splitlines()[1].strip()}" for c in conflicts)

    items, errors, collect_warnings, per_export = collect(exports, len(names), args.keep_empty, args.dedupe)
    warnings.extend(collect_warnings)

    if errors:
        print(f"[错误] 标签校验发现 {len(errors)} 个问题，未写入任何文件：", file=sys.stderr)
        for e in errors[:30]:
            print(f"  - {e}", file=sys.stderr)
        if len(errors) > 30:
            print(f"  ... 另有 {len(errors) - 30} 条", file=sys.stderr)
        return 3

    labeled = sum(1 for it in items if it.records)
    if labeled == 0:
        print("[没有可用数据] 找到了图片目录，但没有任何带标注的图片。", file=sys.stderr)
        print("  先完成标注（X-AnyLabeling 导出 YOLO txt）再重跑。", file=sys.stderr)
        return 2

    train, val = split_items(items, args.val_ratio)
    report = build_report(names, train, val, warnings, errors, exports, per_export)

    print(f"[数据集] {out}")
    print(f"   类别 {len(names)} 个（顺序即 class_id）：")
    for i, n in enumerate(names):
        print(f"     {i:>2}  {n:<20} train={report['per_class']['train'][n]:<5} val={report['per_class']['val'][n]}")
    print(f"   图片：train {len(train)} / val {len(val)}；有标注 {labeled}，背景 {len(items) - labeled}")
    if report["image_sizes"]:
        print(f"   尺寸分布（前几种）：{report['image_sizes']}")
    for w in warnings:
        print(f"   [提示] {w}")
    if args.dry_run:
        print("   [dry-run] 未写任何文件")
        return 0

    stats = write_dataset(out, names, train, val, args.mode, args.force)
    (out / "dataset_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"   写入完成：{dict(stats)}")
    print(f"   下一步：yolo-training\\yolo.cmd cfg=yolo-training/train.yaml data=\"{out / 'data.yaml'}\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
