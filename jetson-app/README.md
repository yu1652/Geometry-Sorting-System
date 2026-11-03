# Jetson Xavier NX 分拣操作台

1024×600 离线操作界面，支持图片和录像回放。核心识别代码已经包含 YOLO 23 类检测、表面掩膜收紧、HSV 颜色识别和 TensorRT 8.5/CUDA Runtime 后端；下一步是把后台识别线程接入“手动识别”按钮。

当前不连接摄像头和串口，不发送 `SORT`、`PUSH` 或其他机械控制指令。

## 当前功能

- 打开 JPG、PNG、BMP、WebP、MP4、AVI 等图片或录像。
- 双视图拼接画面按中线拆分，也支持单视图显示。
- YOLO11 raw 输出解码、23 类 class-aware NMS。
- 表面掩膜恢复并收紧 YOLO 粗框。
- HSV 颜色识别和左右视图结果合并。
- TensorRT 8.5 + CUDA Runtime `ctypes` 后端，不依赖 PyCUDA。
- 保存无框、无文字的原始样本；不会覆盖已有样本。

## 文件职责

- `app.py`：Qt 界面、媒体操作和状态展示。
- `media.py`：真实图片和录像读取，不打开摄像头。
- `core.py`：回放状态与唯一目录存档。
- `vision.py`：letterbox、YOLO raw 解码、NMS、掩膜恢复和 HSV 颜色识别。
- `trt_backend.py`：TensorRT 8.5 + CUDA Runtime `ctypes` 推理后端。
- `vision_runtime.py`：双模型单路识别和左右视图结果合并。
- `vision_config.yaml`：置信度、NMS、掩膜、颜色和类别中文名配置。
- `tests/`：纯逻辑、视觉处理、TensorRT fake 后端和流水线测试。

## Jetson 环境

已确认的目标设备：

```text
Jetson Xavier NX Developer Kit
Ubuntu 20.04.6 LTS
L4T R35.6.5（JetPack 5 系列）
Python 3.8.10
TensorRT 8.5.2.2
```

Windows 的 `.venv` 不能复制到 Jetson。板端使用系统 PyQt5 和 OpenCV：

```bash
sudo apt update
sudo apt install -y python3-opencv python3-pyqt5 python3-venv
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install PyYAML pyserial
```

## 模型文件

模型和 TensorRT Engine 不进入 Git。部署时放入 `models/`：

```text
models/yolo11n_23cls_raw_fp16.engine
models/surface_mask_fp16.engine
models/classes.txt
```

## 运行与验证

Jetson 全屏运行：

```bash
bash run_jetson.sh
```

运行自动测试：

```bash
python -m unittest discover -s tests -v
```

## 下一阶段

将 `vision_runtime.py` 通过长期运行的 Qt 工作线程接入“手动识别”，在 Xavier NX 上用真实 Engine 和样本图片完成验证。通过后再接入三路 USB 摄像头、俯拍标定和机械状态机。
