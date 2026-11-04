# Jetson Xavier NX 分拣操作台

1024×600 离线操作界面，支持图片和录像回放及旧版双摄 26 类识别。模型候选通过清单校验后在识别工作线程试跑；试跑通过仍须操作员明确确认才更新活动指针。

当前不连接摄像头和串口，不发送 `SORT`、`PUSH` 或其他机械控制指令。

## 当前功能

- 打开 JPG、PNG、BMP、WebP、MP4、AVI 等图片或录像。
- 双视图拼接画面按中线拆分，也支持单视图显示。
- YOLO11 raw 输出解码、23 类 class-aware NMS。
- 表面掩膜恢复并收紧 YOLO 粗框。
- HSV 颜色识别和左右视图结果合并。
- TensorRT 8.5 + CUDA Runtime `ctypes` 后端，不依赖 PyCUDA。
- 保存无框、无文字的原始样本；不会覆盖已有样本。
- 侧视检测与表面掩膜使用独立模型槽，保存活动与上一活动版本；默认表面掩膜仍关闭。

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

模型和 TensorRT Engine 不进入 Git。首次部署时将经过校验的旧权重放入 `models/`，在目标板构建 Engine。后续候选存入 `data/model_store/releases/`，须使用同目录 `manifest.json` 声明 hash、形状和预处理版本：

```text
models/SHAPE.onnx
models/shape_legacy_fp16.engine
models/legacy_classes.txt
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

## 旧版双摄标注校验与掩膜导出

在标注环境安装 `pycocotools`，从 `jetson-app` 目录运行：

```bash
python -m pip install pycocotools
python tools/check_legacy_labels.py --images PATH --labels PATH --coco FILE --class-map FILE --output PATH
```

输入为左图在前、右图在后的 640×320 彩色拼接图。COCO 的 `images` 必须记录图像相对路径、宽和高。YOLO TXT 使用相同的相对路径，扩展名改为 `.txt`。每个 COCO 实例使用尺寸为 `[320, 640]` 的整图 RLE。类别映射 JSON 的键为 COCO `category_id`，值为旧模型的 YOLO 类别 ID（0–25），例如 `{"7": 0, "12": 3}`。同侧、同类且有正面积交集的框和掩膜实例必须唯一配对。

输出目录必须尚未存在，父目录须已存在。`images/` 和 `masks/` 中的文件同名，格式为 `图像ID_实例ID_left.png` 或 `图像ID_实例ID_right.png`。彩色半图和单通道掩膜均为 320×320 像素，掩膜前景为 255，背景和通孔为 0。负样本须有 COCO 图像记录及空 YOLO TXT，导出左右半图及两张全零掩膜，文件名中的实例 ID 为 `negative`。校验失败时命令返回非零并指出相关输入。

划分训练、验证和测试集时，仍按实物与采集组划分。同一实物、同一次连拍及同一拼接图的左右半图须留在同组。

## 模型更换与板端交付验证

在 `jetson-app` 目录运行以下桌面验证（Windows）：

```powershell
..\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_model_store.py -v
$env:QT_QPA_PLATFORM='offscreen'
..\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
..\.venv\qt5test\Scripts\python.exe -B -m unittest discover -s tests -p test_legacy_gui.py -v
```

目标板按前述系统依赖创建 system-site-packages 虚拟环境，并在本机执行 `trtexec` 构建 Engine；用 `sha256sum` 记录 Engine 和输入 ONNX 的散列。启动应用后从模型管理选择候选，使用已保存双图试跑并检查识别预览，再显式确认启用。验证独立换掩膜时侧视版本不变；分别试跑成功、失败及回退，并重启核对 `data/model_store/active.json` 的版本。记录 TensorRT、CUDA、JetPack、Engine hash、每个实物的正确/错误/无结果和耗时。手工重点检查类别 ID/中文名、跨中线框、覆膜左右偏移、环件内孔、边界和空场。板端与真实样本验收尚待设备访问；在验收完成前保持覆膜默认关闭且不连接机械输出。

## 下一阶段

在 Xavier NX 上用真实 Engine 和样本图片完成“手动识别”验证。通过后再接入三路 USB 摄像头、俯拍标定和机械状态机。

从已登录的 Jetson 图形桌面或 VNC 终端进入 `jetson-app` 后执行 `bash run_jetson.sh`。仅在 SSH 中设置 `DISPLAY=:0` 可能因图形会话认证失败而无法启动。板端引擎构建和实物验收仍待完成。
