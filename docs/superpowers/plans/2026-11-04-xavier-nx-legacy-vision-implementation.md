# Xavier NX 旧版双摄识别与 COCO 覆膜 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 Xavier NX 上用保存的双摄拼接图或录像验证旧版形状检测、彩色判断和 COCO 表面覆膜，并提供独立换模入口。

**Architecture:** 保留旧模型的拼接输入，分割模型逐半图处理原彩色画面。纯视觉代码返回结构化结果，Qt 后台线程持有推理对象，界面显示框、覆膜和结果。另设离线数据校验入口，COCO 工具只在整理或验收标注的环境使用。

**Tech Stack:** Python 3.8、PyQt5/PySide6、OpenCV、NumPy、PyYAML、TensorRT 8.5.2.2 / CUDA Runtime。离线标注工具另用 pycocotools。

**Spec:** [双摄识别设计](../specs/2026-11-04-xavier-nx-legacy-side-vision-design.md)。

## Global Constraints

- 目标为 Xavier NX、Ubuntu 20.04、Python 3.8.10、1024×600。沿用现有 Qt 先于 OpenCV 的导入顺序。
- 左右各 `320×320`，原彩色拼图 `640×320`，形状模型张量 `[1,3,640,640]` → `[1,30,8400]`。阈值 `0.45`，按类 NMS IoU `0.50`。模型 ID 采用 `SHAPE.onnx` 的 26 类顺序。
- COCO 原图 RLE 尺寸 `[320,640]`，右图偏移 `320`。表面模型按 `[1,3,320,320]` → `[1,1,320,320]` logits 跑左右半图。空、歧义或失败的掩膜有独立状态。
- 本计划交付离线回放与手动识别。三路相机、顶部定位和九点标定、USB 串口、比赛页和比赛计数属于后续计划。首版接入阶段机械输出关闭。
- 按路径提交任务文件。保留现有固件工作区改动，模型、样本图、掩膜及用户活动配置都放在 Git 忽略的目录。
- 测试命令默认在 `jetson-app` 运行。Windows 可用已验证的 `py -3.14 -B`，Jetson 使用 `.venv/bin/python -B`。代码按 Python 3.8 语法编写。

## 文件职责

- `jetson-app/legacy_vision.py` 负责旧版图像变换、识别流程和结构化结果。
- `jetson-app/vision.py` 与 `jetson-app/trt_backend.py` 复用图像变换、解码和 CUDA 资源管理。
- `jetson-app/legacy_vision.yaml` 与 `jetson-app/models/legacy_classes.txt` 固定首版阈值、中文名和类别顺序。
- `jetson-app/tools/check_legacy_labels.py` 读取和校验一套拼图标注，输出左右训练用的彩色图和二值掩膜。
- `jetson-app/vision_worker.py` 持有推理线程、请求和换模生命周期。`jetson-app/app.py` 负责发起操作和显示。
- `jetson-app/model_store.py` 负责模型包校验、活动版本和回退指针。`jetson-app/README.md` 记录运行与板端验收步骤。

### Task 1: 复刻旧版输入并打通 26 类 TensorRT 合同

**Files**

- Create `jetson-app/legacy_vision.py`、`jetson-app/models/legacy_classes.txt`、`jetson-app/tests/test_legacy_vision.py`。
- Modify `jetson-app/vision.py`、`jetson-app/trt_backend.py`、`jetson-app/tests/test_trt_backend.py`。

**Interfaces**

- `prepare_legacy_pair(left_bgr, right_bgr) -> (stitched_bgr, tensor, transform)`。分别接收非空 uint8 BGR 图像，返回新分配的拼图、连续 float32 张量及 `LetterboxTransform`。
- `decode_yolo_raw(output, transform, class_names, confidence_threshold=0.25, iou_threshold=0.7, *, expected_count=23)`。旧版调用显式传 `expected_count=26`。
- `create_backend(path, "yolo_legacy", device=0)`。已存在的 `"yolo"` 和 `"mask"` 合同继续生效。

- [ ] 写测试。构造非方形左右图，并用不同灰度和边缘检验拼接、中心裁剪、上下补边和通道顺序。构造只有 class 25 有分数的旧版输出，确认框解码回原拼图位置。用现有 `FakeEngine(kind="yolo", input_shape=(1,3,640,640), output_shape=(1,30,8400))` 验证 legacy 类型通过，现有默认 yolo 类型拒绝它。

```python
raw = np.zeros((1, 30, 8400), dtype=np.float32)
raw[0, :4, 0] = (400, 210, 20, 10)
raw[0, 4 + 25, 0] = 0.9
result = decode_yolo_raw(raw, transform, classes, 0.45, 0.50, expected_count=26)
assert result[0].class_id == 25
assert result[0].box == (390, 45, 410, 55)
```

- [ ] 运行 `python -B -m unittest discover -s tests -p "test_legacy_vision.py" -v`，确认新入口尚缺时失败。
- [ ] 实现以下预处理核心。每路对中心方形裁剪做 `cv2.INTER_AREA` 缩放。类表取旧训练资料的 26 类且逐索引和可信 SHAPE ONNX 元数据核对，首尾为 `san_leng_zhu`、`ban_qiu_ke`。类别加载对 26 类拒绝重复和保留占位。TensorRT 在分配前检查 `images` 绑定和完整静态维度，继续复用现有设备拷贝和所有权检查。

```python
stitched = cv2.hconcat([left_320, right_320])
gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
edge = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 60, 160)
tensor, transform = letterbox_rgb(cv2.merge([gray, clahe, edge]), size=640)
```

- [ ] 重跑新增用例和 `python -B -m unittest discover -s tests -q`。另核对备份的真实 `SHAPE.onnx` 第 25 类及拼接后的抽样张量。
- [ ] 逐路径提交 `feat: add legacy stitched detector contract`。

### Task 2: 连接形状、颜色和表面掩膜

**Files**

- Modify `jetson-app/legacy_vision.py`、`jetson-app/tests/test_legacy_vision.py`。
- Create `jetson-app/legacy_vision.yaml`。

**Interfaces**

- `LegacyRecognizer(detector, segmenter, class_names, config)`。detector / segmenter 使用 `infer(tensor) -> ndarray` 与 `close()`，segmenter 允许为 `None`。
- `recognize_pair(left_bgr, right_bgr) -> LegacyResult`。结果含 `detection`（可空）、`color`（可空）、`mask_status`（左右各一）、`preview_bgr`、`elapsed_ms`。
- `close()` 从构造、推理所在的同一线程调用。设备后端由持有者统一关闭。

- [ ] 写 fake 后端用例。左右各给一个检测，置信度较高者决定形状。让右半掩膜含一个洞，检查彩色叠层只在右半的前景变化、洞内保持原像素。检查无检测、空掩膜、多候选和分割异常时的状态，确保原图未修改。

```python
assert result.detection.class_id == 25
np.testing.assert_array_equal(result.preview_bgr[120, 440], source[120, 440])
assert np.any(result.preview_bgr[100, 430] != source[100, 430])
np.testing.assert_array_equal(left_input, left_before)
```

- [ ] 运行 `python -B -m unittest discover -s tests -p "test_legacy_vision.py" -v`，确认结果或入口缺失导致失败。
- [ ] 实现。按框与两半幅的横向重叠选侧，平手归左。用各侧最高置信度框做颜色候选，已知色优先，随后比较 HSV 置信度和检测置信度。从旧配置迁移颜色阈值，并用旧软件样本比较输出。覆膜在有明确单个物块的半图上调用 `letterbox_rgb(half, 320)` 和 `restore_surface_mask(logits, transform, threshold=0.5)`。掩膜区域与同侧框须有交集，歧义则跳过该侧覆膜。绘制用拼图副本，先覆膜后画矩形。检测失败返回错误，掩膜失败保留有效形状结果。
- [ ] 重跑新增用例及全套单测。用至少一张带通孔的双图检查掩膜绘制前后前景与洞内像素。
- [ ] 逐路径提交 `feat: combine legacy detections with surface overlays`。

### Task 3: 校验 YOLO 和 COCO 标注并导出掩膜训练样本

**Files**

- Create `jetson-app/tools/check_legacy_labels.py`、`jetson-app/tests/test_legacy_labels.py`。
- Modify `jetson-app/README.md`。

**Interfaces**

- `validate_pair(bgr, yolo_rows, coco_annotations, coco_to_yolo) -> list`。输出按实例的左右半图和等尺寸 uint8 掩膜，前景值 `255`。
- CLI `python tools/check_legacy_labels.py --images PATH --labels PATH --coco FILE --class-map FILE --output PATH`。`--class-map` 为 COCO category ID 到 26 类 YOLO ID 的显式 JSON 映射。输出目录需新建，数据校验失败返回非零并指明图像与实例。

- [ ] 写双实例测试。拼接图 `640×320`，左右各一个物块。构造右侧环形掩膜，编码为全图 RLE。验证解码后左右没有交换，右侧切片不偏移，内孔为 0。另覆盖缺标、类别错配、空掩膜、坐标跨缝和一个矩形匹配多个实例的输入。

```python
full = np.zeros((320, 640), dtype=np.uint8)
full[90:130, 420:460] = 1
full[105:115, 435:445] = 0
encoded = mask_api.encode(np.asfortranarray(full))
decoded = mask_api.decode(encoded)
assert decoded[110, 440] == 0
assert decoded[:, 320:][95, 105] == 1
```

- [ ] 在标注环境装好 `pycocotools` 后运行 `python -B -m unittest discover -s tests -p "test_legacy_labels.py" -v`。普通无标注工具的运行环境跳过该测试文件的 COCO 用例。
- [ ] 实现。按 COCO 的 `image_id` 查找图像，确认高宽和文件名。逐实例解码成整图，检查前景非空且只落一半。从掩膜重算绝对框和面积。将归一化 YOLO 矩形还原到拼图像素，同半图、同类且有交集的候选须唯一。拒绝不匹配的记录，按同一视图裁出彩色图及掩膜，用 `image_id` 和 `annotation_id` 保留来源。负样本须在 COCO 的 `images` 有记录且有空 TXT，导出两张半图和两个全零掩膜。README 写明导出数据仍按实物与采集组划分，左右半图保持同组。
- [ ] 跑新增用例及全套单测。用一对用户标注样本跑 CLI，检查原图、TXT、JSON 未被改动。若新样本尚未交付，使用测试生成的拼图执行同一 CLI，交付后补用户样本验收。
- [ ] 逐路径提交 `feat: validate stitched YOLO and COCO annotations`。

### Task 4: 接入现有「手动识别」和覆膜预览

**Files**

- Create `jetson-app/vision_worker.py`、`jetson-app/tests/test_legacy_gui.py`。
- Modify `jetson-app/app.py`、`jetson-app/tests/smoke_gui.py`。

**Interfaces**

- `VisionWorker(recognizer_factory)` 为后台 QObject。`recognize(request_id, frame_revision, left_bgr, right_bgr)` 发出 `completed(request_id, frame_revision, result)` 或 `failed(request_id, frame_revision, message)`。
- `Window(recognizer_factory=None)` 允许注入 fake。GUI 通过 queued signal 投递，使用独立结果画面或临时叠层显示，`self.frame` 继续保存无标注原图。

- [ ] 写 Qt offscreen 测试。加载程序生成的双图，点击按钮后 fake 接收两张原图。验证识别时回放暂停、重复请求受限。切换源后投递上一帧结果，确认界面丢弃它。保存后用 `cv2.imdecode` 比较像素。关窗时确认后端 close 在推理线程且线程有序退出。

```python
assert window.result_label.text() == expected_shape
np.testing.assert_array_equal(window.frame, original_bgr)
assert fake.infer_thread_id == fake.close_thread_id
```

- [ ] 设置 `QT_QPA_PLATFORM=offscreen` 并运行 `python -B -m unittest discover -s tests -p "test_legacy_gui.py" -v`。确认新 UI 入口尚缺时失败。
- [ ] 实现。Qt5 用 `pyqtSignal/pyqtSlot`，Qt6 用 `Signal/Slot`。默认 factory 在 worker 内从 `models/shape_legacy_fp16.engine`、`models/legacy_classes.txt`、`legacy_vision.yaml` 加载形状识别，掩膜可加载 `models/surface_mask_fp16.engine`。加载媒体前、下一帧和布局变更时更新 `frame_revision`，接收结果前同时比较帧版本和请求 ID。单图模式提示需要双摄拼图。无模型显示明确原因，界面可继续打开文件。关窗先忽略本次关闭事件并异步排队清理，worker 发完成信号，UI 调用 quit，待 thread.finished 后重新关闭窗口，避免 UI 等待自身队列。更新旧 smoke 的占位断言，给测试注入临时样本路径。

```python
if self.closing or request_id != self.active_request_id:
    return
self.active_request_id = None
if frame_revision != self.frame_revision:
    return
# Update the result views without calling display() or replacing self.frame.
```
- [ ] 运行新增 offscreen 用例、回放测试和全套单测，在 1024×600 检查按钮、两路画面、结果及覆膜状态。
- [ ] 逐路径提交 `feat: connect manual recognition to the Qt worker`。

### Task 5: 实现模型更换、回退与板端交付验证

**Files**

- Create `jetson-app/model_store.py`、`jetson-app/tests/test_model_store.py`。
- Modify `jetson-app/vision_worker.py`、`jetson-app/app.py`、`jetson-app/README.md`、`jetson-app/models/README.md`。

**Interfaces**

- `ModelStore(root)` 保存活动集合及上一集合。`validate_manifest(path, slot) -> ModelRelease` 校验候选。`commit_active(active, previous)` 用同目录原子替换。
- Worker 在同一线程执行 `activate_candidate(manifest_path, slot, fixture_pair)` 和 `rollback(fixture_pair)`。形状槽 `side_detector`，掩膜槽 `surface_mask`。两槽各有独立版本。

- [ ] 写故障用例。用临时目录建立活动版本，尝试错误 SHA、错误类别顺序、跨目录文件名及预热抛错。比较失败前后 `active.json` 内容。验证只替换掩膜时侧视版本一致，成功回退后重启读取的仍是已回退的版本。

```python
before = store.active_path.read_bytes()
with self.assertRaises(ValueError):
    store.validate_manifest(candidate_manifest, "side_detector")
self.assertEqual(store.active_path.read_bytes(), before)
```

- [ ] 运行 `python -B -m unittest discover -s tests -p "test_model_store.py" -v`，确认入口尚缺时失败。
- [ ] 实现。模型包由 `manifest.json` 与同目录 engine 组成，必填键为 `schema`（`1`）、`slot`、`version`、`engine`、`sha256`、`input_shape`、`output_shape`、`preprocess`。预处理编号为 `legacy_stitched_v1` 或 `surface_rgb320_v1`。形状槽另需 `class_names` 数组，逐项与官方 26 类顺序核对。只接受同目录普通文件，拒绝绝对路径、路径穿越及链接。应用只在离线空闲时受理切换，并锁住重复切换与推理。候选复制进 `data/model_store/releases`，worker 在同线程反序列化并试跑选定样本，通过后用以下方式发布完整活动指针。加载或写入失败时保留活动版本，失败回执显示原因。回退使用同一校验流程。启动时优先复核已保存的活动版本，只有首次运行且无活动记录时使用 Task 4 的本地默认路径。UI 分两个槽显示版本、状态、耗时及「选择模型」「试跑并启用」「回退」。

```python
with temporary_path.open("w", encoding="utf-8") as stream:
    json.dump({"active": active, "previous": previous}, stream, ensure_ascii=False)
    stream.flush()
    os.fsync(stream.fileno())
os.replace(temporary_path, active_path)
```
- [ ] 跑新增测试、Qt offscreen 冒烟和全套单测。板端将已核对的旧 `SHAPE.onnx` 放入本地 models 目录，在机械输出关闭的情况下构建 Engine。用保存的双图及掩膜模型验证「手动识别」、独立换模和回退。记录每件实物的正确、错误、无结果和耗时。对照旧程序原始类别编号，另核对中文名。覆膜重点检查环件内孔、左右对齐、边缘与空场。若现有 `surface_mask_fp16.engine` 未通过，交付版覆膜需等新训练权重完成验证。

```bash
cd /home/nvidia/jetson_sorter_app
/usr/src/tensorrt/bin/trtexec --onnx=models/SHAPE.onnx --saveEngine=models/shape_legacy_fp16.engine --fp16 --workspace=256 --avgTiming=1 --buildOnly
bash run_jetson.sh
```

- [ ] README 记录板端实测环境、引擎和标注验收。逐路径提交 `feat: add independent vision model updates`。

## 下一阶段接口

离线验收后，采集服务把同一物块的左右帧送入同一识别入口。顶部定位服务单独处理俯拍画面及九点 XY。调试页与比赛页共用一次识别结果和设备状态。自动控制单独验收人工对零、调试升降、坐标边界、断线恢复和 `DONE` 计数后再启用。
