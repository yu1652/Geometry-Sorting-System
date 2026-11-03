# Geometry Sorting System

几何体智能分拣系统，包含 ESP32 机构控制、YOLO 训练工具和 Jetson Xavier NX 离线识别程序。

## 目录

- `src/`：ESP32 XY 机构、舵机和推杆控制固件。
- `yolo-training/`：23 类 YOLO 数据整理、训练与导出脚本。
- `jetson-app/`：Jetson Xavier NX 图形界面、TensorRT 推理、表面掩膜和 HSV 颜色识别。

模型权重、TensorRT Engine、训练数据、虚拟环境和密钥不会提交到仓库；部署时按各目录的 README 单独放置。

## 当前安全边界

Jetson 程序当前只进行离线图片/录像识别，不发送 `SORT`、`PUSH` 或其他机械指令。接入三路 USB 摄像头、坐标标定和 ESP32 串口控制前，需要先在实机完成独立验收。
