# Xavier NX 补图 24 类

从本次发布包更新板端的 `jetson-app`。保留板端的 `data/`、相机绑定和实机标定。

| 目录 | 发布包内的 ONNX 源文件 | 板端生成的 TensorRT FP16 引擎 |
| --- | --- | --- |
| `models/butu_20261110/` | `shape_detector.onnx` | `shape_detector_fp16.engine` |
| `models/butu_20261110/` | `surface_mask.onnx` | `surface_mask_fp16.engine` |
| `models/linux_20261107/` | `tray_object_detector.onnx` | `tray_object_detector_fp16.engine` |

两个模型目录都应包含发布包提供的 `onnx-sha256.txt`。构建前会逐一校验补图版的形状模型、表面掩膜，以及 20261107 版的托盘模型。补图模式只复用后者的托盘模型，板上可以只安装上表这三个 ONNX。形状模型输入为 `[1,3,320,320]`，输出为 `[1,28,2100]`。掩膜源模型的批次维度可变，板端引擎使用单张 320×320 输入。

在 Xavier NX 上使用随 JetPack 安装的 TensorRT 构建。从 `jetson-app` 目录运行：

```bash
bash deploy/build_models.sh --butu-24
.venv/bin/python deploy/smoke_butu.py
bash run_jetson.sh --butu-24 --offline --preprocessed --file samples/butu_24_half_cylinder.png
```

脚本会保留已有非空引擎。更换 TensorRT 版本或模型后，先将相应的旧引擎移到备份位置，再重新构建。共享的托盘引擎也供 `--linux-20261107` 使用。构建失败时，临时产物留在脚本报告的 `.engine-build.*` 目录，可检查后重试。

预检只运行三个 TensorRT 引擎，不打开串口和相机。随包半圆柱样本是已处理特征拼图，用于核对形状和掩膜。识别实际颜色时用原始彩色双摄图，并将启动参数改为 `--raw`。先用留存的双相机图和空区画面离线验证。核对 24 类标签顺序、检测框和表面掩膜边界，确认孔洞、阴影和背景不会被误判为物块。随后从板端打开实机模式，先核对相机画面、托盘定位、九点标定及空区参考。

```bash
bash run_jetson.sh --butu-24
```

沿用现有 ESP32 0.1.0 固件。确认机构在手动原点且行程内无人、无障碍后，按现场流程做空载检查，再逐件核对识别、推料坐标、箱号和投放确认。发现异常时停止，保留调试日志或诊断包以便定位。Windows 上的模拟构建只验证了命令和文件处理，TensorRT 构建、模型推理和运动验收均需在板端完成。
