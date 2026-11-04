# Xavier NX model files

The legacy side detector is the verified 26-class `SHAPE.onnx` lineage. Build `shape_legacy_fp16.engine` on the target Xavier NX with its installed TensorRT; neither it nor source weights are committed. Keep `legacy_classes.txt` as the authoritative ordered class list. The baseline `surface_mask_fp16.engine` remains disabled until separately validated.

Model updates use `data/model_store/releases/<slot>-<version>/manifest.json` plus its same-directory engine. Manifests must declare schema 1, slot, version, engine filename, SHA-256, input/output shapes and the slot's preprocessing ID. Side detector manifests also include `class_names` in the exact `legacy_classes.txt` order. Import candidates while offline, run a saved left/right sample, inspect the preview, then explicitly confirm activation. A failed validation or trial leaves the active model selected. Retain prior release directories for rollback; never overwrite an existing version directory.

Generate and verify the legacy engine on target hardware:

```bash
/usr/src/tensorrt/bin/trtexec --onnx=models/SHAPE.onnx --saveEngine=models/shape_legacy_fp16.engine --fp16 --workspace=256 --avgTiming=1 --buildOnly
sha256sum models/shape_legacy_fp16.engine
```

Do not commit `.pt`, `.onnx`, `.engine` files or sample data. Real-device build and recognition acceptance are pending; no motion output is connected.

Manifest keys are `schema` (1), `slot` (`side_detector` or `surface_mask`), `version`, `engine`, `sha256`, `input_shape`, `output_shape`, and `preprocess` (`legacy_stitched_v1` or `surface_rgb320_v1` respectively). The side detector also requires `class_names`. The engine must be a regular file in the manifest directory.
