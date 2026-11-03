import sys
import tempfile
import unittest
from pathlib import Path
from typing import Tuple, get_type_hints

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vision import (  # noqa: E402 - local module under test
    LetterboxTransform,
    classify_hsv,
    decode_yolo_raw,
    letterbox_rgb,
    load_class_names,
    restore_surface_mask,
    tighten_box,
)


CLASS_NAMES = ["class_%d" % index for index in range(23)]
COLOR_CONFIG = {
    "unknown_color": "未知颜色",
    "min_color_confidence": 0.35,
    "min_valid_pixels": 50,
    "black_value_max": 45,
    "white_value_min": 165,
    "achromatic_saturation_max": 35,
    "hsv_ranges": {
        "红色": [
            {"lower": [0, 80, 50], "upper": [10, 255, 255]},
            {"lower": [170, 80, 50], "upper": [180, 255, 255]},
        ],
        "绿色": [{"lower": [35, 60, 40], "upper": [85, 255, 255]}],
        "蓝色": [{"lower": [100, 60, 40], "upper": [130, 255, 255]}],
    },
}


class LetterboxTests(unittest.TestCase):
    def test_letterbox_converts_bgr_to_rgb_and_tracks_inverse_geometry(self):
        """Catches swapped BGR channels and incorrect top/bottom padding offsets."""
        frame = np.zeros((2, 4, 3), dtype=np.uint8)
        frame[0, 0] = [10, 20, 30]

        tensor, transform = letterbox_rgb(frame, size=8)

        self.assertEqual(tensor.shape, (1, 3, 8, 8))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(tensor.flags.c_contiguous)
        self.assertEqual(transform, LetterboxTransform(4, 2, 8, 2.0, 0, 2, 8, 4))
        np.testing.assert_allclose(tensor[0, :, 2, 0], [30 / 255.0, 20 / 255.0, 10 / 255.0])
        np.testing.assert_allclose(tensor[0, :, 0, 0], [114 / 255.0] * 3)

    def test_letterbox_rejects_empty_or_non_bgr_frames(self):
        """Catches model calls on empty captures or malformed image arrays."""
        with self.assertRaises(ValueError):
            letterbox_rgb(np.empty((0, 4, 3), dtype=np.uint8))
        with self.assertRaises(ValueError):
            letterbox_rgb(np.zeros((4, 4), dtype=np.uint8))


class DecodeTests(unittest.TestCase):
    def setUp(self):
        self.transform = LetterboxTransform(100, 50, 320, 3.2, 0, 80, 320, 160)

    def test_decode_raw_yolo11_has_no_objectness_and_restores_clipped_boxes(self):
        """Catches treating a class score as obsolete YOLO objectness confidence."""
        raw = np.zeros((1, 27, 1), dtype=np.float32)
        raw[0, :4, 0] = [160, 160, 64, 64]
        raw[0, 4 + 3, 0] = 0.9

        detections = decode_yolo_raw(raw, self.transform, CLASS_NAMES)

        self.assertEqual(len(detections), 1)
        detection = detections[0]
        self.assertEqual((detection.class_id, detection.class_name), (3, "class_3"))
        self.assertAlmostEqual(detection.confidence, 0.9)
        self.assertEqual(detection.box, (40.0, 15.0, 60.0, 35.0))

    def test_decode_nms_only_suppresses_overlaps_in_the_same_class(self):
        """Catches cross-class NMS dropping a valid second object and unstable order."""
        raw = np.zeros((1, 27, 3), dtype=np.float32)
        raw[0, :4, 0] = [160, 160, 64, 64]
        raw[0, 4 + 3, 0] = 0.90
        raw[0, :4, 1] = [161, 160, 64, 64]
        raw[0, 4 + 3, 1] = 0.80
        raw[0, :4, 2] = [161, 160, 64, 64]
        raw[0, 4 + 4, 2] = 0.95

        detections = decode_yolo_raw(raw, self.transform, CLASS_NAMES)

        self.assertEqual([(item.class_id, round(item.confidence, 2)) for item in detections], [(4, 0.95), (3, 0.90)])

    def test_decode_rejects_bad_class_layout_and_nonfinite_model_output(self):
        """Catches silently decoding a model with the wrong class head or NaN scores."""
        with self.assertRaises(ValueError):
            decode_yolo_raw(np.zeros((1, 26, 1), dtype=np.float32), self.transform, CLASS_NAMES)
        raw = np.zeros((1, 27, 1), dtype=np.float32)
        raw[0, 4, 0] = np.nan
        with self.assertRaises(ValueError):
            decode_yolo_raw(raw, self.transform, CLASS_NAMES)
        with self.assertRaises(ValueError):
            decode_yolo_raw(np.zeros((1, 27, 1), dtype=np.float32), self.transform, CLASS_NAMES[:-1])

    def test_decode_uses_actual_rounded_resize_scale_for_noninteger_letterbox_ratio(self):
        """Catches y coordinates drifting when rounded resize height differs from scale."""
        _, transform = letterbox_rgb(np.zeros((2, 3, 3), dtype=np.uint8), size=8)
        raw = np.zeros((1, 27, 1), dtype=np.float32)
        raw[0, :4, 0] = [4.0, 3.5, 4.0, 3.0]
        raw[0, 4, 0] = 0.9

        detections = decode_yolo_raw(raw, transform, CLASS_NAMES)

        self.assertAlmostEqual(transform.scale_x, 8.0 / 3.0)
        self.assertAlmostEqual(transform.scale_y, 2.5)
        self.assertEqual(detections[0].box, (0.75, 0.4, 2.25, 1.6))

    def test_decode_accepts_confidence_equal_to_threshold(self):
        """Catches a strict score comparison dropping detections at the configured cutoff."""
        raw = np.zeros((1, 27, 1), dtype=np.float32)
        raw[0, :4, 0] = [160, 160, 64, 64]
        raw[0, 4, 0] = 0.25
        self.assertEqual(len(decode_yolo_raw(raw, self.transform, CLASS_NAMES)), 1)

    def test_decode_keeps_iou_equal_to_threshold_under_strict_nms(self):
        """Catches equality changing NMS from suppress-only-above to suppress-at-threshold."""
        transform = LetterboxTransform(320, 320, 320, 1.0, 0, 0, 320, 320)
        raw = np.zeros((1, 27, 2), dtype=np.float32)
        raw[0, :4, 0] = [8.5, 10.0, 17.0, 10.0]
        raw[0, 4, 0] = 0.9
        raw[0, :4, 1] = [11.5, 10.0, 17.0, 10.0]
        raw[0, 4, 1] = 0.8

        self.assertEqual(len(decode_yolo_raw(raw, transform, CLASS_NAMES, iou_threshold=0.7)), 2)


class ClassNameTests(unittest.TestCase):
    def test_load_class_names_keeps_only_the_23_runtime_classes(self):
        """Catches reserved training labels leaking into runtime classifier indexes."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "classes.txt"
            path.write_text("\n".join(CLASS_NAMES + ["reserved_23", "reserved_24", "reserved_25"]), encoding="utf-8")
            self.assertEqual(load_class_names(path), CLASS_NAMES)

    def test_load_class_names_accepts_exactly_23_runtime_entries_without_reserved_tail(self):
        """Catches rejecting a deployed class file that correctly omits training reserves."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "classes.txt"
            path.write_text("\n".join(CLASS_NAMES), encoding="utf-8")
            self.assertEqual(load_class_names(path), CLASS_NAMES)

    def test_load_class_names_rejects_duplicate_reserved_or_unexpected_entries(self):
        """Catches corrupt class files that would shift class IDs without an error."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "classes.txt"
            path.write_text("\n".join(CLASS_NAMES[:-1] + ["class_0"]), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_class_names(path)
            path.write_text("\n".join(CLASS_NAMES + ["foreign_23"]), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_class_names(path)

    def test_load_class_names_rejects_reserved_name_within_first_23(self):
        """Catches a reserved training index replacing a deployed runtime class."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "classes.txt"
            path.write_text("\n".join(CLASS_NAMES[:-1] + ["reserved_23"]), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_class_names(path)

    def test_load_class_names_requires_the_complete_ordered_reserved_tail(self):
        """Catches malformed reserve tails that otherwise conceal a shifted labels file."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "classes.txt"
            for tail in (
                ["reserved_23"],
                ["reserved_23", "reserved_25"],
                ["reserved_24", "reserved_23", "reserved_25"],
                ["reserved_23", "reserved_24", "reserved_24"],
            ):
                path.write_text("\n".join(CLASS_NAMES + tail), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_class_names(path)


class ConfigTests(unittest.TestCase):
    def test_vision_config_parses_with_literal_thresholds_and_model_class_names(self):
        """Catches a deploy-time YAML key mismatch or model class-name lookup failure."""
        config_path = Path(__file__).resolve().parents[1] / "vision_config.yaml"
        with config_path.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)

        self.assertEqual(config["model"], {
            "input_size": 320,
            "class_count": 23,
            "confidence_threshold": 0.25,
            "iou_threshold": 0.7,
            "mask_threshold": 0.5,
        })
        self.assertEqual(config["recognition"], {
            "unknown_color": "未知颜色",
            "min_color_confidence": 0.35,
        })
        self.assertEqual(config["color"], {
            "hsv_ranges": {
                "红色": [
                    {"lower": [0, 80, 50], "upper": [10, 255, 255]},
                    {"lower": [170, 80, 50], "upper": [180, 255, 255]},
                ],
                "橙色": [{"lower": [10, 80, 50], "upper": [20, 255, 255]}],
                "黄色": [{"lower": [21, 80, 50], "upper": [38, 255, 255]}],
                "绿色": [{"lower": [35, 60, 40], "upper": [85, 255, 255]}],
                "青色": [{"lower": [85, 60, 40], "upper": [100, 255, 255]}],
                "蓝色": [{"lower": [100, 60, 40], "upper": [130, 255, 255]}],
                "紫色": [{"lower": [130, 50, 40], "upper": [160, 255, 255]}],
            },
            "bbox_center_ratio": 0.6,
            "valid_min_value": 40,
            "valid_max_value": 245,
            "valid_min_saturation": 40,
            "min_valid_pixels": 50,
            "black_value_max": 45,
            "white_value_min": 165,
            "achromatic_saturation_max": 35,
            "fallback_to_segmentation": True,
        })
        self.assertEqual(config["class_name_map"], {
            "ba_leng_huan_zhu": "八棱环柱", "ba_leng_zhu": "八棱柱", "ban_qiu": "半球",
            "ban_qiu_ke": "半球壳", "liu_leng_huan_zhu": "六棱环柱", "liu_leng_zhu": "六棱柱",
            "qi_leng_huan_zhu": "七棱环柱", "qi_leng_zhu": "七棱柱", "san_leng_zhu": "三棱柱",
            "shi_leng_huan_zhu": "十棱环柱", "shi_leng_zhu": "十棱柱", "si_leng_huan_zhu": "四棱环柱",
            "si_leng_tai": "四棱台", "si_leng_zhu": "四棱柱", "si_leng_zhu_jia": "四棱柱甲",
            "si_leng_zhui": "四棱锥", "wu_leng_huan_zhu": "五棱环柱", "wu_leng_tai": "五棱台",
            "wu_leng_zhu": "五棱柱", "yuan_huan_zhu": "圆环柱", "yuan_tai": "圆台",
            "yuan_zhu": "圆柱", "yuan_zhui": "圆锥",
        })


class MaskTests(unittest.TestCase):
    def test_tighten_box_annotation_allows_the_preserved_float_fallback(self):
        """Catches a misleading integer-only type contract for empty YOLO float boxes."""
        self.assertEqual(get_type_hints(tighten_box)["return"], Tuple[float, float, float, float])

    def test_restore_surface_mask_discards_letterbox_padding_before_nearest_resize(self):
        """Catches segmentation padding becoming false foreground in the source image."""
        transform = LetterboxTransform(4, 2, 8, 2.0, 0, 2, 8, 4)
        logits = np.full((1, 1, 8, 8), -10.0, dtype=np.float32)
        logits[0, 0, 0:2, :] = 10.0
        logits[0, 0, 2:6, 0:4] = 10.0

        mask = restore_surface_mask(logits, transform)

        expected = np.array([[True, True, False, False], [True, True, False, False]])
        np.testing.assert_array_equal(mask, expected)

    def test_restore_surface_mask_rejects_wrong_or_nonfinite_logits(self):
        """Catches using a detection map as a mask or propagating NaN into geometry."""
        transform = LetterboxTransform(4, 2, 8, 2.0, 0, 2, 8, 4)
        with self.assertRaises(ValueError):
            restore_surface_mask(np.zeros((1, 8, 8), dtype=np.float32), transform)
        logits = np.zeros((1, 1, 8, 8), dtype=np.float32)
        logits[0, 0, 0, 0] = np.inf
        with self.assertRaises(ValueError):
            restore_surface_mask(logits, transform)

    def test_restore_surface_mask_treats_zero_logit_as_background_at_half_threshold(self):
        """Catches sigmoid(0) being included despite the model's strict > 0.5 contract."""
        transform = LetterboxTransform(4, 2, 8, 2.0, 0, 2, 8, 4)
        logits = np.zeros((1, 1, 8, 8), dtype=np.float32)
        np.testing.assert_array_equal(restore_surface_mask(logits, transform), np.zeros((2, 4), dtype=bool))

    def test_tighten_box_falls_back_when_its_mask_intersection_is_empty(self):
        """Catches an empty segmentation result collapsing an otherwise usable detection."""
        mask = np.zeros((5, 5), dtype=bool)
        self.assertEqual(tighten_box((2, 2, 7, 7), mask), (2, 2, 7, 7))

    def test_tighten_box_preserves_an_empty_float_box_exactly(self):
        """Catches floor, ceil, or clipping corrupting a YOLO float fallback box."""
        mask = np.zeros((5, 5), dtype=bool)
        self.assertEqual(tighten_box((1.25, 1.5, 5.75, 6.125), mask), (1.25, 1.5, 5.75, 6.125))

    def test_tighten_box_returns_the_tight_pixel_extent_inside_detection(self):
        """Catches tightening against unrelated mask pixels outside the detection box."""
        mask = np.zeros((6, 6), dtype=bool)
        mask[2, 3] = True
        mask[4, 4] = True
        mask[0, 0] = True
        self.assertEqual(tighten_box((1, 1, 5, 5), mask), (3, 2, 5, 5))


class ColorTests(unittest.TestCase):
    def test_classify_hsv_rejects_horizontal_zero_width_before_rounding(self):
        """Catches a continuous zero-width box expanding to one classified pixel column."""
        frame = np.full((100, 100, 3), [0, 0, 255], dtype=np.uint8)
        result = classify_hsv(
            frame,
            np.ones((100, 100), dtype=bool),
            (20.2, 10.0, 20.2, 90.0),
            COLOR_CONFIG,
        )
        self.assertEqual((result.name, result.confidence, result.valid_pixels), ("未知颜色", 0.0, 0))

    def test_classify_hsv_rejects_vertical_zero_height_before_rounding(self):
        """Catches a continuous zero-height box expanding to one classified pixel row."""
        frame = np.full((100, 100, 3), [0, 0, 255], dtype=np.uint8)
        result = classify_hsv(
            frame,
            np.ones((100, 100), dtype=bool),
            (10.0, 30.4, 90.0, 30.4),
            COLOR_CONFIG,
        )
        self.assertEqual((result.name, result.confidence, result.valid_pixels), ("未知颜色", 0.0, 0))

    def test_classify_hsv_rejects_point_box_before_rounding(self):
        """Catches a continuous point box expanding into a one-pixel ROI."""
        frame = np.full((100, 100, 3), [0, 0, 255], dtype=np.uint8)
        result = classify_hsv(
            frame,
            np.ones((100, 100), dtype=bool),
            (20.2, 10.7, 20.2, 10.7),
            COLOR_CONFIG,
        )
        self.assertEqual((result.name, result.confidence, result.valid_pixels), ("未知颜色", 0.0, 0))

    def test_classify_hsv_sorts_an_inverted_box_that_still_has_area(self):
        """Catches the empty-box guard accidentally rejecting valid inverted coordinates."""
        frame = np.full((100, 100, 3), [0, 0, 255], dtype=np.uint8)
        result = classify_hsv(
            frame,
            np.ones((100, 100), dtype=bool),
            (90.0, 90.0, 10.0, 10.0),
            COLOR_CONFIG,
        )
        self.assertEqual((result.name, result.confidence, result.valid_pixels), ("红色", 1.0, 6400))

    def test_classify_hsv_converts_coordinate_overflow_to_value_error(self):
        """Catches huge integer coordinates leaking an inconsistent OverflowError."""
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "box coordinates must be numeric"):
            classify_hsv(frame, np.ones((10, 10), dtype=bool), (10 ** 400, 0, 1, 1), COLOR_CONFIG)

    def test_classify_hsv_rejects_nonfinite_box_coordinates(self):
        """Catches NaN or infinity reaching integer ROI conversion."""
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        for coordinate in (np.nan, np.inf, -np.inf):
            with self.subTest(coordinate=coordinate):
                with self.assertRaisesRegex(ValueError, "box coordinates must be finite"):
                    classify_hsv(frame, np.ones((10, 10), dtype=bool), (coordinate, 0, 1, 1), COLOR_CONFIG)

    def test_classify_hsv_returns_unknown_for_a_completely_outside_float_box(self):
        """Catches float fallback coordinates being used directly as NumPy slice indices."""
        frame = np.full((10, 10, 3), [0, 0, 255], dtype=np.uint8)
        result = classify_hsv(
            frame,
            np.ones((10, 10), dtype=bool),
            (12.5, 1.25, 15.75, 7.5),
            COLOR_CONFIG,
        )
        self.assertEqual((result.name, result.confidence, result.valid_pixels), ("未知颜色", 0.0, 0))

    def test_classify_hsv_clips_a_partially_outside_float_box_before_sampling(self):
        """Catches edge detections sampling outside the frame instead of their clipped ROI."""
        frame = np.full((10, 10, 3), [0, 0, 255], dtype=np.uint8)
        result = classify_hsv(
            frame,
            np.ones((10, 10), dtype=bool),
            (-4.5, 0.2, 8.3, 9.7),
            COLOR_CONFIG,
        )
        self.assertEqual((result.name, result.confidence, result.valid_pixels), ("红色", 1.0, 90))

    def test_classify_hsv_accepts_the_split_recognition_and_color_config(self):
        """Catches production YAML nesting causing a missing confidence setting."""
        full_config = {
            "recognition": {
                "unknown_color": "未知颜色",
                "min_color_confidence": 0.35,
            },
            "color": dict(COLOR_CONFIG, unknown_color=None, min_color_confidence=None),
        }
        del full_config["color"]["unknown_color"]
        del full_config["color"]["min_color_confidence"]
        frame = np.full((10, 10, 3), [0, 0, 255], dtype=np.uint8)

        result = classify_hsv(frame, np.ones((10, 10), dtype=bool), (0, 0, 10, 10), full_config)

        self.assertEqual((result.name, result.confidence, result.valid_pixels), ("红色", 1.0, 100))

    def test_classify_hsv_handles_both_red_hue_wraparound_ranges(self):
        """Catches red pixels near hue 180 being missed by a single hue interval."""
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        frame[:5, :] = [0, 0, 255]
        frame[5:, :] = [8, 0, 255]
        result = classify_hsv(frame, np.ones((10, 10), dtype=bool), (0, 0, 10, 10), COLOR_CONFIG)
        self.assertEqual((result.name, result.valid_pixels), ("红色", 100))
        self.assertAlmostEqual(result.confidence, 1.0)

    def test_classify_hsv_uses_value_and_saturation_for_black_white_and_gray(self):
        """Catches achromatic objects being forced into an arbitrary chromatic hue."""
        mask = np.ones((10, 10), dtype=bool)
        for bgr, expected in (([20, 20, 20], "黑色"), ([220, 220, 220], "白色"), ([100, 100, 100], "灰色")):
            frame = np.full((10, 10, 3), bgr, dtype=np.uint8)
            result = classify_hsv(frame, mask, (0, 0, 10, 10), COLOR_CONFIG)
            self.assertEqual((result.name, result.confidence, result.valid_pixels), (expected, 1.0, 100))

    def test_classify_hsv_returns_unknown_for_insufficient_or_low_majority_pixels(self):
        """Catches noisy tiny masks and mixed-color regions being reported as confident."""
        frame = np.full((10, 10, 3), [0, 0, 255], dtype=np.uint8)
        sparse_mask = np.zeros((10, 10), dtype=bool)
        sparse_mask.flat[:49] = True
        sparse = classify_hsv(frame, sparse_mask, (0, 0, 10, 10), COLOR_CONFIG)
        self.assertEqual((sparse.name, sparse.confidence, sparse.valid_pixels), ("未知颜色", 0.0, 49))

        frame.reshape(-1, 3)[:34] = [0, 0, 255]
        frame.reshape(-1, 3)[34:67] = [0, 255, 0]
        frame.reshape(-1, 3)[67:] = [255, 0, 0]
        mixed = classify_hsv(frame, np.ones((10, 10), dtype=bool), (0, 0, 10, 10), COLOR_CONFIG)
        self.assertEqual((mixed.name, mixed.confidence, mixed.valid_pixels), ("未知颜色", 0.34, 100))


if __name__ == "__main__":
    unittest.main()
