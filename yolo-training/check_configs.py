#!/usr/bin/env python3
"""静态校验 yolo-training/ 下的四个 YAML，防止拼错参数名被 Ultralytics 静默忽略。

检查内容：
  classes.yaml      非空、唯一、只用 [a-z0-9_]
  data.example.yaml 结构合法；不写 path / nc；names 与 classes.yaml 一致
  train.yaml        每个键都是 Ultralytics 合法参数；mode/task/imgsz 等关键值正确
  export.yaml       每个键都是合法参数；mode/format/imgsz/batch/nms 与部署端契约一致

用法：py -3.14 yolo-training/check_configs.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
NAME_RE = re.compile(r"^[a-z0-9_]+$")
DATA_KEYS = {"path", "train", "val", "test", "names", "nc", "download", "channels"}

problems: list[str] = []
notes: list[str] = []


def fail(msg: str) -> None:
    problems.append(msg)


def ok(msg: str) -> None:
    notes.append(msg)


def load(name: str) -> dict:
    path = HERE / name
    if not path.is_file():
        fail(f"缺文件：{path}")
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        fail(f"{name} 不是合法 YAML：{exc}")
        return {}
    if not isinstance(data, dict):
        fail(f"{name} 顶层必须是映射")
        return {}
    return data


def check_classes() -> list[str]:
    data = load("classes.yaml")
    names = data.get("names")
    if not isinstance(names, list) or not names:
        fail("classes.yaml: 需要一个非空的 names 列表")
        return []
    names = [str(n) for n in names]
    if len(set(names)) != len(names):
        fail(f"classes.yaml: 类别名重复 {len(names) - len(set(names))} 个")
    bad = [n for n in names if not NAME_RE.match(n)]
    if bad:
        fail(f"classes.yaml: 非法类别名 {bad}（只允许英文小写/数字/下划线）")
    ok(f"classes.yaml: {len(names)} 类 —— {', '.join(names)}")
    return names


def check_data_example(names: list[str]) -> None:
    data = load("data.example.yaml")
    unknown = set(data) - DATA_KEYS
    if unknown:
        fail(f"data.example.yaml: 未知键 {sorted(unknown)}")
    if "path" in data:
        fail("data.example.yaml: 不应该写 path（会让数据集根不可移植）")
    if "nc" in data:
        fail("data.example.yaml: 不应该写 nc（会与 names 长度漂移，交给自动推导）")
    for key in ("train", "val"):
        if not isinstance(data.get(key), str):
            fail(f"data.example.yaml: 缺少字符串类型的 {key}:")
    if names and data.get("names") != names:
        fail("data.example.yaml: names 与 classes.yaml 不一致（文档漂移）")
    ok("data.example.yaml: 结构正常，train/val 相对路径，names 与类别表一致")


def cfg_keys() -> set[str]:
    try:
        from ultralytics.utils import DEFAULT_CFG_DICT
    except Exception as exc:  # noqa: BLE001
        fail(f"无法导入 ultralytics 默认参数表：{exc}（先 pip install ultralytics）")
        return set()
    return set(DEFAULT_CFG_DICT)


def check_train(valid: set[str]) -> None:
    data = load("train.yaml")
    if not data:
        return
    unknown = set(data) - valid
    if unknown:
        fail(f"train.yaml: 不是合法的 Ultralytics 参数 {sorted(unknown)}")
    if data.get("mode") != "train":
        fail(f"train.yaml: mode 必须是 train（现在是 {data.get('mode')!r}；缺省会跑成 predict）")
    if data.get("task") != "detect":
        fail("train.yaml: task 必须是 detect")
    if data.get("imgsz") != 640:
        fail(f"train.yaml: imgsz 应为 640 以匹配部署端 ONNX（现在 {data.get('imgsz')!r}）")
    if "data" in data:
        fail("train.yaml: 不要写 data，由命令行 data=... 传入")
    for key in ("epochs", "batch", "workers", "patience"):
        if key in data and not isinstance(data[key], int):
            fail(f"train.yaml: {key} 应为整数")
    for key in ("lr0", "lrf"):
        if key in data and not isinstance(data[key], (int, float)):
            fail(f"train.yaml: {key} 应为数字")
    for key in ("hsv_h", "hsv_s", "hsv_v", "translate", "scale", "mosaic", "mixup", "cutmix", "fliplr", "flipud"):
        if key in data and not isinstance(data[key], (int, float)):
            fail(f"train.yaml: {key} 应为数字")
    degree = data.get("degrees", 0)
    if not isinstance(degree, (int, float)) or not 0 <= degree <= 180:
        fail(f"train.yaml: degrees 应在 0~180（现在 {degree!r}）")
    ok(
        "train.yaml: "
        f"imgsz={data.get('imgsz')} batch={data.get('batch')} device={data.get('device')!r} "
        f"epochs={data.get('epochs')} model={data.get('model')}"
    )


def check_export(valid: set[str]) -> None:
    data = load("export.yaml")
    if not data:
        return
    unknown = set(data) - valid
    if unknown:
        fail(f"export.yaml: 不是合法的 Ultralytics 参数 {sorted(unknown)}")
    expected = {
        "mode": "export",
        "format": "onnx",
        "imgsz": 640,
        "batch": 1,
        "nms": False,
        "simplify": True,
        "dynamic": False,
    }
    for key, want in expected.items():
        if data.get(key) != want:
            fail(f"export.yaml: {key} 应为 {want!r}（现在是 {data.get(key)!r}）")
    ok("export.yaml: ONNX 原始检测头（nms=False）、640 输入、batch=1")


def main() -> int:
    valid = cfg_keys()
    names = check_classes()
    check_data_example(names)
    check_train(valid)
    check_export(valid)

    print("=" * 66)
    for n in notes:
        print(f"  [OK]   {n}")
    for p in problems:
        print(f"  [FAIL] {p}")
    print("=" * 66)
    if problems:
        print(f"结果：FAIL（{len(problems)} 个问题）")
        return 1
    print(f"结果：PASS（{len(notes)} 项检查，全部通过）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
