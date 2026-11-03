# 服务器训练手册（AutoDL / Linux）

## 环境（已确认）

| 项 | 值 |
|---|---|
| Python | `/root/miniconda3/bin/python`（3.12.3，**不在 PATH 上，要用全路径**） |
| PyTorch | 2.8.0+cu128，torchvision 0.23.0+cu128 |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition，97887 MiB，驱动 595.104.02 |
| ultralytics | 8.4.138（与本地校验版本一致） |
| 磁盘 | 系统盘 30G；数据盘 `/root/autodl-tmp` 50G |

## 目录

```text
/root/autodl-tmp/ylers/
├── yolo-training/     配置套件（classes.yaml / build_dataset.py / train.yaml / export.yaml …）
├── datasets/          数据集（构建输出，如 datasets/shape）
├── datasets/incoming/ 你从标注设备传上来的原始 session / 压缩包
├── runs/              训练输出
└── weights/           预训练权重（yolov8n.pt）
```

## 常用命令

先进入项目目录：

```bash
cd /root/autodl-tmp/ylers
PY=/root/miniconda3/bin/python
YOLO=/root/miniconda3/bin/yolo
```

自检配置：

```bash
$PY yolo-training/check_configs.py
```

检查标注端导出的类别是否与预设一致（**数据到手先跑这个**）：

```bash
$PY yolo-training/inspect_sessions.py --sessions datasets/incoming
```

合成数据集（多 session 合并 + 校验 + 切分）：

```bash
$PY yolo-training/build_dataset.py --sessions datasets/incoming --out datasets/shape
```

训练（`cfg=` 必须写在最前面）：

```bash
$YOLO cfg=yolo-training/train.yaml data=datasets/shape/data.yaml model=weights/yolov8n.pt
```

导出 ONNX（给部署端用的原始检测头，不带 NMS）：

```bash
$YOLO cfg=yolo-training/export.yaml model=runs/sort/shape/weights/best.pt
```

核对导出结果：

```bash
$PY - <<'PY'
import onnxruntime as ort
s = ort.InferenceSession("runs/sort/shape/weights/best.onnx", providers=["CPUExecutionProvider"])
print(s.get_modelmeta().custom_metadata_map.get("names"))
print(s.get_inputs()[0].shape, s.get_outputs()[0].shape)
PY
```

## 注意

- 容器里 `python` 不在 PATH，`nproc` 显示的是宿主机核数（208），实际配额是 22 vCPU；
  训练参数里 `workers: 8` 已经合适，不要按 `nproc` 调大。
- 数据放在 `/root/autodl-tmp`，不要放系统盘（只有 30G）。
- 关机前确认权重已拷出，实例释放后 `/root/autodl-tmp` 会一起没。
