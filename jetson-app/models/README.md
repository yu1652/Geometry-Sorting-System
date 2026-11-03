# 模型目录

`classes.txt` 是运行时类别顺序，进入 Git。模型权重和 TensorRT Engine 不进入 Git。

在 Xavier NX 上放置：

```text
yolo11n_23cls_raw_fp16.engine
surface_mask_fp16.engine
```

这两个 Engine 必须在目标 Jetson/TensorRT 环境中生成。不要提交 `.pt`、`.onnx` 或 `.engine` 文件。
