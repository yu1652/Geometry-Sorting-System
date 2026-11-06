# Xavier NX model files

The legacy side detector is the verified 26-class `SHAPE.onnx` lineage. Build `shape_legacy_fp16.engine` on the target Xavier NX with its installed TensorRT; neither it nor source weights are committed. Keep `legacy_classes.txt` as the authoritative ordered class list. The supplied `surface_mask_fp16.engine` now loads by default when present. Its training images are processed feature halves, so the local baseline uses `surface_mask.input_mode: legacy_features`. Missing engines and unusable masks are reported explicitly; validate surface boundaries and holes on real captures before automatic sorting.

Model updates use `data/model_store/releases/<slot>-<version>/manifest.json` plus its same-directory engine. Manifests must declare schema 1, slot, version, engine filename, SHA-256, input/output shapes and the slot's preprocessing ID. Side detector manifests also include `class_names` in the exact `legacy_classes.txt` order. Import candidates while offline, run a saved left/right sample, inspect the preview, then explicitly confirm activation. A failed validation or trial leaves the active model selected. Retain prior release directories for rollback; never overwrite an existing version directory.

Generate and verify the legacy engine on target hardware:

```bash
/usr/src/tensorrt/bin/trtexec --onnx=models/SHAPE.onnx --saveEngine=models/shape_legacy_fp16.engine --fp16 --workspace=256 --avgTiming=1 --buildOnly
sha256sum models/shape_legacy_fp16.engine
```

Do not commit `.pt`, `.onnx`, `.engine` files or sample data. Real-device build and recognition acceptance are pending; no motion output is connected.

Manifest keys are `schema` (1), `slot` (`side_detector` or `surface_mask`), `version`, `engine`, `sha256`, `input_shape`, `output_shape`, and `preprocess`. Use `legacy_stitched_v1` for the side detector. The supplied feature-trained mask uses `surface_features320_v1`; a future mask trained on original RGB halves uses `surface_rgb320_v1`. Both mask contracts consume RGB NCHW float32 320x320 /255, but their image contents differ. Release metadata selects the mask input independently of the local baseline. A raw-RGB mask cannot run on feature-only files. The side detector also requires `class_names`. The engine must be a regular file in the manifest directory.

Releases are immutable. Correcting the preprocessing declaration for an already installed engine requires a new `version`. Reimporting the same version/hash with different metadata is rejected; it never silently reuses the earlier input contract or overwrites an installed manifest.

The separate overhead locator uses the archived single-class `tray_object_detector.onnx` with input `[1,3,640,640]` and output `[1,5,8400]`. For later bench acceptance, place the ONNX file in `models/` and build on the Xavier NX:

```bash
/usr/src/tensorrt/bin/trtexec --onnx=models/tray_object_detector.onnx --saveEngine=models/tray_object_detector_fp16.engine --fp16 --workspace=256 --avgTiming=1 --buildOnly
```

Verify predictions on saved overhead frames before connecting the locator to motion. The simulated console uses a fixed sample detection and has no model-file requirement.
