# Xavier NX 和 ESP32 首次上板验收

此版本包含离线识别、录像演练和整机仿真。默认启动界面可验证形状模型，当前现场版仍需把三路实时取图、模型和 USB 串口接进同一控制流程。将部署通过和整机分拣通过分别记录。

## 1. Jetson 部署

确认目标是 Xavier NX，已登录 Ubuntu 图形桌面。将部署包中的 `jetson-app` 解压到新目录，再进入该目录。保留原程序与原标定以便回退。

```bash
sudo apt update
sudo apt install -y python3-venv python3-opencv python3-pyqt5 python3-yaml python3-serial
python3 -m venv --system-site-packages .venv
bash deploy/build_models.sh
bash run_jetson.sh
```

本地部署包包含项目提供的 `SHAPE.onnx`、`tray_object_detector.onnx`、`surface_mask.onnx` 和校验值。形状模型可在离线界面用原始双摄拼接图点「手动识别」。用俯拍样本检查顶部模型，现场检查表面掩膜能否保留环件通孔后，再在模型管理中启用。引擎脚本保留已有非空 Engine。切换 ONNX 或设备环境时，在新目录构建对应 Engine。

## 2. ESP32

Windows 端使用现场确认的固件源码。当前本地另有移除出口闸门舵机的改动，GitHub 版本仍包含出口闸门逻辑。先根据接线确认用哪一版，烧录前断开电机和舵机驱动电源，避免启动初始化引起动作。

```powershell
pio device list
pio run -e esp32dev
pio run -e esp32dev -t upload --upload-port COMx
```

在选定的 PlatformIO 工程目录运行。把 `COMx` 改为实际枚举的 ESP32 端口，关闭占用端口的串口监视器后再上传。烧录完成后，在执行器仍断电时，以 115200 打开串口，核对 `PING` → `PONG`、`VERSION?` 和 `STATUS?`，保存输出。

## 3. 整机开始前

1. 核对三路相机的角色、画幅和稳定设备路径，核对旧版标定点在实际图中的位置和六箱绑定。
2. 在现场操作员确认执行器断电且机构安全的情况下手动放到物理零点。无原点限位时，`RESET` 只把当前位置计为软件零点。
3. 确认实机上位机已通过空载握手、行程、抬降、停机测试，再按单件、连续多件的顺序验收。固件阻塞动作期间，软件 STOP 不能代替现场断电或实体急停。
