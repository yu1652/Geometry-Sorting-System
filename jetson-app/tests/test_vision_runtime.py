import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np
import yaml

from vision_runtime import OfflineRecognizer


CLASS_NAMES = ["class_%d" % index for index in range(23)]


def runtime_config():
    return {
        "model": {
            "input_size": 320,
            "class_count": 23,
            "confidence_threshold": 0.25,
            "iou_threshold": 0.7,
            "mask_threshold": 0.5,
        },
        "recognition": {
            "unknown_color": "未知颜色",
            "min_color_confidence": 0.35,
        },
        "color": {
            "hsv_ranges": {
                "红色": [
                    {"lower": [0, 80, 50], "upper": [10, 255, 255]},
                    {"lower": [170, 80, 50], "upper": [180, 255, 255]},
                ],
                "绿色": [
                    {"lower": [35, 60, 40], "upper": [85, 255, 255]}
                ],
            },
            "min_valid_pixels": 1,
            "black_value_max": 45,
            "white_value_min": 165,
            "achromatic_saturation_max": 35,
        },
        "class_name_map": {"class_3": "三号工件"},
    }


class FakeBackend:
    def __init__(self, output):
        self.output = output
        self.inputs = []
        self.close_count = 0

    def infer(self, tensor):
        self.inputs.append(tensor)
        return self.output.copy()

    def close(self):
        self.close_count += 1


class SequenceBackend(FakeBackend):
    def __init__(self, outputs):
        super().__init__(None)
        self.outputs = list(outputs)

    def infer(self, tensor):
        self.inputs.append(tensor)
        return self.outputs[len(self.inputs) - 1].copy()


def raw_detection(class_id, confidence):
    raw = np.zeros((1, 27, 1), dtype=np.float32)
    raw[0, :4, 0] = [160, 160, 100, 100]
    raw[0, 4 + class_id, 0] = confidence
    return raw


def full_mask(foreground=True):
    value = 10.0 if foreground else -10.0
    return np.full((1, 1, 320, 320), value, dtype=np.float32)


class RuntimeFixture(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.yolo_path = self.root / "yolo.engine"
        self.mask_path = self.root / "mask.engine"
        self.classes_path = self.root / "classes.txt"
        self.config_path = self.root / "vision_config.yaml"
        self.yolo_path.write_bytes(b"yolo")
        self.mask_path.write_bytes(b"mask")
        self.classes_path.write_text("\n".join(CLASS_NAMES), encoding="utf-8")
        self.config_path.write_text(
            yaml.safe_dump(runtime_config(), allow_unicode=True), encoding="utf-8"
        )

    def make_recognizer(self, factory):
        return OfflineRecognizer(
            yolo_engine_path=self.yolo_path,
            mask_engine_path=self.mask_path,
            class_names_path=self.classes_path,
            config_path=self.config_path,
            backend_factory=factory,
        )

    def make_sequence_recognizer(self, yolo_outputs, mask_outputs):
        yolo = SequenceBackend(yolo_outputs)
        mask = SequenceBackend(mask_outputs)
        recognizer = self.make_recognizer(
            lambda path, model_kind, device=0: {"yolo": yolo, "mask": mask}[model_kind]
        )
        self.addCleanup(recognizer.close)
        return recognizer


class BackendLifecycleTests(RuntimeFixture):
    def test_constructor_is_native_lazy_and_first_recognition_creates_both_backends(self):
        """Catches TensorRT/CUDA creation moving back onto the UI constructor thread."""
        calls = []
        yolo = FakeBackend(np.zeros((1, 27, 1), dtype=np.float32))
        mask = FakeBackend(np.full((1, 1, 320, 320), -10.0, dtype=np.float32))

        def factory(path, model_kind, device=0):
            calls.append((Path(path), model_kind, device))
            return {"yolo": yolo, "mask": mask}[model_kind]

        recognizer = self.make_recognizer(factory)
        self.addCleanup(recognizer.close)
        self.assertEqual(calls, [])

        result = recognizer.recognize_view(np.zeros((20, 20, 3), dtype=np.uint8))

        self.assertFalse(result.detected)
        self.assertEqual(
            calls,
            [(self.yolo_path, "yolo", 0), (self.mask_path, "mask", 0)],
        )

    def test_second_backend_creation_failure_closes_first_without_masking_error(self):
        """Catches a leaked YOLO CUDA context when mask-engine construction fails."""
        yolo = FakeBackend(np.zeros((1, 27, 1), dtype=np.float32))

        def factory(path, model_kind, device=0):
            if model_kind == "mask":
                raise RuntimeError("mask factory failed")
            return yolo

        recognizer = self.make_recognizer(factory)

        with self.assertRaisesRegex(RuntimeError, "mask factory failed"):
            recognizer.recognize_view(np.zeros((20, 20, 3), dtype=np.uint8))

        self.assertEqual(yolo.close_count, 1)
        recognizer.close()
        self.assertEqual(yolo.close_count, 1)

    def test_close_is_idempotent_for_both_owned_backends(self):
        """Catches double-free paths when UI shutdown invokes close more than once."""
        yolo = FakeBackend(np.zeros((1, 27, 1), dtype=np.float32))
        mask = FakeBackend(np.full((1, 1, 320, 320), -10.0, dtype=np.float32))
        recognizer = self.make_recognizer(
            lambda path, model_kind, device=0: {"yolo": yolo, "mask": mask}[model_kind]
        )
        recognizer.recognize_view(np.zeros((20, 20, 3), dtype=np.uint8))

        recognizer.close()
        recognizer.close()

        self.assertEqual((yolo.close_count, mask.close_count), (1, 1))

    def test_close_from_non_owner_thread_preserves_backends_for_owner_cleanup(self):
        """Catches a wrong-thread close marking resources closed before CUDA rejects it."""
        yolo = FakeBackend(np.zeros((1, 27, 1), dtype=np.float32))
        mask = FakeBackend(full_mask(False))
        recognizer = self.make_recognizer(
            lambda path, model_kind, device=0: {"yolo": yolo, "mask": mask}[model_kind]
        )
        recognizer.recognize_view(np.zeros((20, 20, 3), dtype=np.uint8))
        failures = []

        worker = threading.Thread(target=lambda: self._capture_close(recognizer, failures))
        worker.start()
        worker.join()

        self.assertEqual(len(failures), 1)
        self.assertIn("owner thread", str(failures[0]))
        self.assertEqual((yolo.close_count, mask.close_count), (0, 0))
        recognizer.close()
        self.assertEqual((yolo.close_count, mask.close_count), (1, 1))

    @staticmethod
    def _capture_close(recognizer, failures):
        try:
            recognizer.close()
        except BaseException as exc:
            failures.append(exc)

    def test_concurrent_first_use_creates_only_one_backend_pair(self):
        """Catches check-then-create races leaking a duplicate native backend pair."""
        yolo = FakeBackend(np.zeros((1, 27, 1), dtype=np.float32))
        mask = FakeBackend(full_mask(False))
        workers_ready = threading.Barrier(2)
        yolo_factories_ready = threading.Barrier(2)
        calls = []
        results = []
        failures = []

        def factory(path, model_kind, device=0):
            calls.append(model_kind)
            if model_kind == "yolo":
                try:
                    yolo_factories_ready.wait(timeout=0.5)
                except threading.BrokenBarrierError:
                    pass
            return {"yolo": yolo, "mask": mask}[model_kind]

        recognizer = self.make_recognizer(factory)

        def worker():
            try:
                workers_ready.wait(timeout=2.0)
                results.append(
                    recognizer.recognize_view(
                        np.zeros((20, 20, 3), dtype=np.uint8)
                    )
                )
                recognizer.close()
            except BaseException as exc:
                failures.append(exc)

        first = threading.Thread(target=worker)
        second = threading.Thread(target=worker)
        first.start()
        second.start()
        first.join(timeout=2.0)
        second.join(timeout=2.0)

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(calls, ["yolo", "mask"])
        self.assertEqual(len(results), 1)
        self.assertEqual(len(failures), 1)
        self.assertRegex(str(failures[0]), "owner thread|closed")
        self.assertEqual((yolo.close_count, mask.close_count), (1, 1))


class SingleViewPipelineTests(RuntimeFixture):
    def test_one_contiguous_tensor_is_reused_without_modifying_source_frame(self):
        """Catches duplicate preprocessing, non-contiguous model input, and source drawing."""
        yolo = FakeBackend(np.zeros((1, 27, 1), dtype=np.float32))
        mask = FakeBackend(np.full((1, 1, 320, 320), -10.0, dtype=np.float32))
        recognizer = self.make_recognizer(
            lambda path, model_kind, device=0: {"yolo": yolo, "mask": mask}[model_kind]
        )
        self.addCleanup(recognizer.close)
        frame = np.full((20, 40, 3), [7, 11, 13], dtype=np.uint8)
        original = frame.copy()

        recognizer.recognize_view(frame)

        self.assertIs(yolo.inputs[0], mask.inputs[0])
        self.assertTrue(yolo.inputs[0].flags.c_contiguous)
        self.assertEqual(yolo.inputs[0].shape, (1, 3, 320, 320))
        np.testing.assert_array_equal(frame, original)

    def test_primary_detection_uses_mask_tightening_hsv_and_class_mapping(self):
        """Catches selecting model order over confidence or sampling color outside the mask."""
        raw = np.zeros((1, 27, 2), dtype=np.float32)
        raw[0, :4, 0] = [160, 160, 100, 100]
        raw[0, 4 + 4, 0] = 0.80
        raw[0, :4, 1] = [160, 160, 100, 100]
        raw[0, 4 + 3, 1] = 0.90
        logits = np.full((1, 1, 320, 320), -10.0, dtype=np.float32)
        logits[0, 0, 120:200, 130:190] = 10.0
        yolo = FakeBackend(raw)
        mask = FakeBackend(logits)
        recognizer = self.make_recognizer(
            lambda path, model_kind, device=0: {"yolo": yolo, "mask": mask}[model_kind]
        )
        self.addCleanup(recognizer.close)
        frame = np.zeros((320, 320, 3), dtype=np.uint8)
        frame[120:200, 130:190] = [0, 0, 255]

        result = recognizer.recognize_view(frame)

        self.assertTrue(result.detected)
        self.assertEqual((result.class_id, result.raw_class_name), (3, "class_3"))
        self.assertEqual(result.class_name, "三号工件")
        self.assertAlmostEqual(result.detection_confidence, 0.90)
        self.assertEqual(result.box, (130, 120, 190, 200))
        self.assertEqual(result.color_name, "红色")
        self.assertAlmostEqual(result.color_confidence, 1.0)

    def test_missing_class_mapping_falls_back_to_raw_name(self):
        """Catches an unmapped deployment class becoming an unusable blank UI label."""
        raw = np.zeros((1, 27, 1), dtype=np.float32)
        raw[0, :4, 0] = [160, 160, 100, 100]
        raw[0, 4 + 4, 0] = 0.90
        logits = np.full((1, 1, 320, 320), 10.0, dtype=np.float32)
        yolo = FakeBackend(raw)
        mask = FakeBackend(logits)
        recognizer = self.make_recognizer(
            lambda path, model_kind, device=0: {"yolo": yolo, "mask": mask}[model_kind]
        )
        self.addCleanup(recognizer.close)

        result = recognizer.recognize_view(
            np.full((320, 320, 3), [0, 0, 255], dtype=np.uint8)
        )

        self.assertEqual(result.class_name, "class_4")

    def test_no_detection_is_normal_and_annotation_owns_its_data(self):
        """Catches no-object frames raising and annotation aliasing the camera buffer."""
        yolo = FakeBackend(np.zeros((1, 27, 1), dtype=np.float32))
        mask = FakeBackend(np.full((1, 1, 320, 320), -10.0, dtype=np.float32))
        recognizer = self.make_recognizer(
            lambda path, model_kind, device=0: {"yolo": yolo, "mask": mask}[model_kind]
        )
        self.addCleanup(recognizer.close)
        frame = np.zeros((20, 20, 3), dtype=np.uint8)

        result = recognizer.recognize_view(frame)

        self.assertFalse(result.detected)
        self.assertIsNone(result.class_name)
        self.assertEqual(result.color_name, "未知颜色")
        self.assertFalse(np.shares_memory(result.annotated_frame, frame))
        self.assertTrue(result.annotated_frame.flags.owndata)


class MultiViewConsensusTests(RuntimeFixture):
    @staticmethod
    def red_frame():
        return np.full((320, 320, 3), [0, 0, 255], dtype=np.uint8)

    @staticmethod
    def green_frame():
        return np.full((320, 320, 3), [0, 255, 0], dtype=np.uint8)

    def test_no_detected_view_reports_no_object(self):
        """Catches an empty scene being mislabeled as a single-view result."""
        recognizer = self.make_sequence_recognizer(
            [np.zeros((1, 27, 1), dtype=np.float32)], [full_mask(False)]
        )

        result = recognizer.recognize_views([self.red_frame()])

        self.assertEqual(result.status, "未识别到物体")
        self.assertIsNone(result.final_class_name)
        self.assertIsNone(result.detection_confidence)

    def test_only_detected_view_becomes_single_view_result(self):
        """Catches a usable result being discarded because its peer has no detection."""
        recognizer = self.make_sequence_recognizer(
            [raw_detection(3, 0.8), np.zeros((1, 27, 1), dtype=np.float32)],
            [full_mask(True), full_mask(False)],
        )

        result = recognizer.recognize_views([self.red_frame(), self.red_frame()])

        self.assertEqual(result.status, "单路识别")
        self.assertEqual(result.final_class_name, "三号工件")
        self.assertEqual(result.final_color_name, "红色")
        self.assertAlmostEqual(result.detection_confidence, 0.8)
        self.assertEqual(len(result.views), 2)

    def test_two_matching_classes_and_colors_average_confidence(self):
        """Catches successful consensus retaining only one view's confidence."""
        recognizer = self.make_sequence_recognizer(
            [raw_detection(3, 0.8), raw_detection(3, 0.6)],
            [full_mask(True), full_mask(True)],
        )

        result = recognizer.recognize_views([self.red_frame(), self.red_frame()])

        self.assertEqual(result.status, "识别成功")
        self.assertEqual((result.final_class_name, result.final_color_name), ("三号工件", "红色"))
        self.assertAlmostEqual(result.detection_confidence, 0.7)

    def test_class_conflict_has_no_mechanical_final_class(self):
        """Catches disagreeing cameras emitting an actionable class command."""
        recognizer = self.make_sequence_recognizer(
            [raw_detection(3, 0.8), raw_detection(4, 0.6)],
            [full_mask(True), full_mask(True)],
        )

        result = recognizer.recognize_views([self.red_frame(), self.red_frame()])

        self.assertEqual(result.status, "结果不确定")
        self.assertIsNone(result.final_class_name)

    def test_explicit_color_conflict_has_no_mechanical_final_class(self):
        """Catches matching shapes bypassing a red-versus-green disagreement."""
        recognizer = self.make_sequence_recognizer(
            [raw_detection(3, 0.8), raw_detection(3, 0.6)],
            [full_mask(True), full_mask(True)],
        )

        result = recognizer.recognize_views([self.red_frame(), self.green_frame()])

        self.assertEqual(result.status, "结果不确定")
        self.assertIsNone(result.final_class_name)
        self.assertEqual(result.final_color_name, "未知颜色")

    def test_unknown_color_yields_to_the_other_explicit_color(self):
        """Catches an unknown view erasing a compatible explicit color consensus."""
        recognizer = self.make_sequence_recognizer(
            [raw_detection(3, 0.8), raw_detection(3, 0.6)],
            [full_mask(False), full_mask(True)],
        )

        result = recognizer.recognize_views([self.red_frame(), self.red_frame()])

        self.assertEqual(result.status, "识别成功")
        self.assertEqual(result.final_color_name, "红色")
        self.assertEqual(result.final_class_name, "三号工件")


class ValidationAndErrorTests(RuntimeFixture):
    def test_invalid_frame_is_rejected_before_native_backend_creation(self):
        """Catches malformed camera data needlessly allocating TensorRT/CUDA resources."""
        calls = []
        recognizer = self.make_recognizer(
            lambda *args, **kwargs: calls.append((args, kwargs))
        )

        with self.assertRaisesRegex(ValueError, "frame_bgr"):
            recognizer.recognize_view(np.zeros((20, 20), dtype=np.uint8))

        self.assertEqual(calls, [])

    def test_recognize_views_requires_exactly_one_or_two_frames(self):
        """Catches empty, oversized, or accidentally unwrapped frame collections."""
        recognizer = self.make_recognizer(lambda *args, **kwargs: None)
        for frames in ([], [None, None, None], np.zeros((20, 20, 3), dtype=np.uint8)):
            with self.subTest(type=type(frames), length=len(frames)):
                with self.assertRaisesRegex(ValueError, "one or two"):
                    recognizer.recognize_views(frames)

    def test_missing_runtime_files_name_the_missing_resource(self):
        """Catches deployment failures surfacing later as opaque backend errors."""
        paths = {
            "yolo_engine_path": self.yolo_path,
            "mask_engine_path": self.mask_path,
            "class_names_path": self.classes_path,
            "config_path": self.config_path,
        }
        cases = (
            ("yolo_engine_path", "YOLO engine"),
            ("mask_engine_path", "surface-mask engine"),
            ("class_names_path", "class names"),
            ("config_path", "vision config"),
        )
        for key, message in cases:
            with self.subTest(resource=key):
                arguments = dict(paths)
                arguments[key] = self.root / (key + ".missing")
                with self.assertRaisesRegex(FileNotFoundError, message):
                    OfflineRecognizer(**arguments, backend_factory=lambda *args: None)

    def test_config_missing_required_model_key_is_a_clear_constructor_error(self):
        """Catches a KeyError occurring only after native backends were allocated."""
        config = runtime_config()
        del config["model"]["mask_threshold"]
        self.config_path.write_text(
            yaml.safe_dump(config, allow_unicode=True), encoding="utf-8"
        )

        with self.assertRaisesRegex(ValueError, "mask_threshold"):
            self.make_recognizer(lambda *args, **kwargs: None)

    def test_malformed_nested_color_config_fails_before_backend_creation(self):
        """Catches malformed HSV ranges and mappings failing only after CUDA allocation."""
        cases = []
        missing_lower = runtime_config()
        del missing_lower["color"]["hsv_ranges"]["红色"][0]["lower"]
        cases.append((missing_lower, "lower"))
        bad_channel_count = runtime_config()
        bad_channel_count["color"]["hsv_ranges"]["红色"][0]["upper"] = [10, 255]
        cases.append((bad_channel_count, "three"))
        bad_mapping = runtime_config()
        bad_mapping["class_name_map"]["class_3"] = None
        cases.append((bad_mapping, "class_name_map"))

        for config, message in cases:
            with self.subTest(message=message):
                self.config_path.write_text(
                    yaml.safe_dump(config, allow_unicode=True), encoding="utf-8"
                )
                calls = []
                with self.assertRaisesRegex(ValueError, message):
                    self.make_recognizer(
                        lambda *args, **kwargs: calls.append((args, kwargs))
                    )
                self.assertEqual(calls, [])

    def test_wrong_model_output_shapes_are_reported_by_model_name(self):
        """Catches YOLO and mask contract errors becoming indistinguishable downstream."""
        cases = (
            (np.zeros((1, 26, 1), dtype=np.float32), full_mask(False), "YOLO.*shape"),
            (np.zeros((1, 27, 1), dtype=np.float32), np.zeros((1, 320, 320), dtype=np.float32), "mask.*shape"),
        )
        for yolo_output, mask_output, message in cases:
            with self.subTest(message=message):
                yolo = FakeBackend(yolo_output)
                mask = FakeBackend(mask_output)
                recognizer = self.make_recognizer(
                    lambda path, model_kind, device=0: {"yolo": yolo, "mask": mask}[model_kind]
                )
                with self.assertRaisesRegex(ValueError, message):
                    recognizer.recognize_view(np.zeros((20, 20, 3), dtype=np.uint8))
                recognizer.close()

    def test_backend_inference_exception_is_not_wrapped_or_swallowed(self):
        """Catches loss of actionable TensorRT failure details at the runtime boundary."""
        class FailingBackend(FakeBackend):
            def infer(self, tensor):
                raise RuntimeError("TensorRT enqueue detail 47")

        yolo = FailingBackend(None)
        mask = FakeBackend(full_mask(False))
        recognizer = self.make_recognizer(
            lambda path, model_kind, device=0: {"yolo": yolo, "mask": mask}[model_kind]
        )
        self.addCleanup(recognizer.close)

        with self.assertRaisesRegex(RuntimeError, "TensorRT enqueue detail 47"):
            recognizer.recognize_view(np.zeros((20, 20, 3), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
