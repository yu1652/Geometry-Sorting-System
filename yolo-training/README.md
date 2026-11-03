# YOLO 训练配置套件（YLERS 双相机采集 → Ultralytics）

还没拍照、还没标注也能先用：类别数量你自己定，改 `classes.yaml` 一处，其余全部自动跟随。

## 文件

| 文件 | 用途 |
|---|---|
| `classes.yaml` | **类别表，唯一真源**。行序 = `class_id`，改数量/种类只改这里 |
| `build_dataset.py` | 扫 session 导出 → 校验标签 → 分层切 train/val → 生成 `data.yaml` |
| `check_configs.py` | 静态自检 4 个 YAML，防止参数名拼错被 Ultralytics 静默忽略 |
| `train.yaml` | 训练参数（GPU 机用），用法 `yolo cfg=train.yaml ...` |
| `export.yaml` | ONNX 导出参数，锁死与部署端一致的 640 输入 + 原始检测头 |
| `data.example.yaml` | `data.yaml` 的讲解范例（真正用哪份由 `build_dataset.py` 生成） |

## 0. 环境

本机已验证：Python 3.14 + `ultralytics 8.4.138`，Torch 是 **CPU 版**（`2.14.0+cpu`，CUDA 不可用）。
所以本机只适合校验配置和跑 1~3 个 epoch 的流程冒烟，真训练请放到有 CUDA 的机器上（训练机同样要求 Ultralytics 8.4.x）。

本机有两个环境事实，所以下面的命令统一用 `py -3.14` 和 `yolo-training\yolo.cmd`：

- PATH 上的 `python` 是 Microsoft Store 占位程序，直接敲会报错；用 `py -3.14` 可以；
- `yolo.exe` 虽然装了，但不在 PATH 上，直接敲 `yolo` 会提示"不是内部或外部命令"。

```powershell
py -3.14 -m pip install -U ultralytics pyyaml
```

## 1. 先定类别（重要）

打开 `classes.yaml`，`names` 列表就是类别，行号 0、1、2… 就是 YOLO 的 `class_id`。

- 现预置 **26 类**（棱柱/环柱/筒架/锥/台/球六组，按棱数升序编号），行尾注释写中文名；
- 改类别数量只改这个列表，别插在中间：**新类别追加到末尾**，否则已有标签的 id 会整体错位；
- **标注之前必须定下来**：标完再改顺序，所有 `.txt` 里的整数 id 会整体错位，只能重新导出标签；
- 类别名用**汉语拼音**（小写 + 下划线分词，如 `si_leng_huan_zhu`），不用中文、不带声调；

改完先自检：

```powershell
py -3.14 yolo-training\check_configs.py
```

## 2. 采集 + 标注

1. **采集**：数据集工作室选 `single_yolo` 模式，分辨率档与部署端保持一致（部署端 `imgsz=640`）。
   每次采集会落一个 session 目录，导出结构形如：

   ```text
   ...\ylers_dualcam_dataset\dataset\sessions\session_20260528_220342\yolo_dataset\
       images\  labels\  data.yaml
   ```

2. **标注**：X-AnyLabeling 导出 **YOLO 格式 txt**（每行 `class_id cx cy w h`，坐标 0~1 归一化），
   类别下拉顺序必须与 `classes.yaml` 一致。

3. 还没数据时，可以先看目标目录结构：

   ```powershell
   py -3.14 yolo-training\build_dataset.py --init --out D:\datasets\sort\shape
   ```

## 3. 合成数据集

把多个 session 合并成一份训练集（图片会复制到输出目录，原 session 不动）：

```powershell
py -3.14 yolo-training\build_dataset.py ^
  --sessions "D:\captures\dataset\sessions" ^
  --out "D:\datasets\sort\shape"
```

输出：

```text
D:\datasets\sort\shape\
    data.yaml                 # 自动生成，不要手改
    dataset_report.json       # 每类数量、尺寸分布、跳过/警告明细
    images\train\  images\val\
    labels\train\  labels\val\
```

常用参数：

| 参数 | 作用 |
|---|---|
| `--dry-run` | 只统计，不写文件 |
| `--drop-empty` | 丢弃无标签图（默认保留为背景负样本） |
| `--dedupe` | 按内容哈希去掉重复图片 |
| `--mode symlink` | 用软链接代替复制（省空间，需要管理员/开发者模式） |
| `--force` | 重跑时覆盖上一次本工具生成的内容 |
| `--force-names` | 强行忽略 session 自带类别表与 `classes.yaml` 的顺序冲突（默认拒绝） |
| `--check` | 只校验已有数据集，不写文件 |

**`build_dataset.py` 会拦住三类错误**：标签列数/坐标越界、`class_id` 超出类别表、多个 session 类别顺序不一致。

## 4. 训练

`cfg=` **必须写在最前面**：Ultralytics 解析到 `cfg=` 时会用它整体替换之前累积的参数，写在它前面的参数会被丢掉。

```powershell
yolo-training\yolo.cmd cfg=yolo-training\train.yaml data=D:\datasets\sort\shape\data.yaml
```

- 默认 `model=yolov8n.pt`（与部署端现有的 YOLOv8n 系列同一代）；没有网络时改 `yolov8n.yaml` 从头训；
- `imgsz=640` 不要改，必须和部署端 ONNX 输入一致；
- 显存不够：命令行追加 `batch=8`；本机 CPU 调试：

  ```powershell
  yolo-training\yolo.cmd cfg=yolo-training\train.yaml data=... device=cpu workers=2 batch=4 imgsz=320 epochs=3
  ```

结果在 `runs/sort/shape/weights/best.pt`。

## 5. 导出 ONNX

```powershell
yolo-training\yolo.cmd cfg=yolo-training\export.yaml model=runs\sort\shape\weights\best.pt
```

导出后**必须核对元数据**（部署端靠原始检测头自己解码，不能带 NMS）：

```python
import onnxruntime as ort
s = ort.InferenceSession("runs/sort/shape/weights/best.onnx", providers=["CPUExecutionProvider"])
print(s.get_modelmeta().custom_metadata_map["names"])   # 顺序应与 classes.yaml 一致
print(s.get_inputs()[0].shape)                          # [1, 3, 640, 640]
print(s.get_outputs()[0].shape)                         # [1, 4+nc, 8400]
```

不满足就检查是不是 `nms=True` 或 `imgsz` 被改过。

## 6. 回填到部署端

把 `best.onnx` 拷成推送程序目录下的 `weights/shape_detector.onnx`（或改它 `config.yaml` 里的 `yolo.model_path`）。
这一步属于部署程序，本套件不改动那边的任何文件。

## 常见问题

**`data.yaml` 为什么没有 `path` 和 `nc`？**
没有 `path` 时 Ultralytics 以 `data.yaml` 所在目录为数据集根，整个目录搬到 Linux 训练机仍然有效；
没有 `nc` 时 `nc` 自动等于 `names` 长度，改类别不会出现两者不一致的报错。

**报错 "dataset not found" / 找到的是 `C:\Users\你\datasets`？**
说明 `data.yaml` 里被手写了相对 `path`。删掉 `path`，让 `data.yaml` 和 `images/` `labels/` 保持同级。

**`val` 里一张图都没有？**
样本太少时分层切分会保底补足；如果总图片数 < 5，建议先多采一点，否则验证指标没有意义。

**标注框位置全错 / 训练 loss 不降？**
先看 `dataset_report.json` 的 `image_sizes`：确认训练用的是双拼图还是单视图，且与部署时喂给模型的图一致。

**`'yolo' 不是内部或外部命令`？**
`yolo.exe` 不在 PATH。用本套件里的 `yolo-training\yolo.cmd`（等价于 `yolo`），
或把 `C:\Users\<你>\AppData\Local\Python\pythoncore-3.14-64\Scripts` 加进 PATH。

**Windows 下中文乱码？**
输出重定向时可能乱码，不影响文件内容；需要的话设置 `$env:PYTHONIOENCODING="utf-8"`。
