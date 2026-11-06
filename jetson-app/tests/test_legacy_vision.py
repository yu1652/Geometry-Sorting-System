import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from legacy_vision import LegacyRecognizer, prepare_legacy_pair
from vision import decode_yolo_raw, load_class_names


CLASSES_FILE = Path(__file__).resolve().parents[1] / "models" / "legacy_classes.txt"


class LegacyVisionTests(unittest.TestCase):
    def test_preprocessed_pair_preserves_features_without_running_enhancement_again(self):
        from unittest.mock import patch
        import cv2
        left = np.full((320, 320, 3), (43, 87, 5), dtype=np.uint8)
        right = np.full((320, 320, 3), (71, 112, 19), dtype=np.uint8)
        with patch.object(cv2, 'createCLAHE', side_effect=AssertionError('double preprocessing')):
            stitched, tensor, transform = prepare_legacy_pair(left, right, preprocessed=True)
        np.testing.assert_array_equal(stitched, np.hstack((left, right)))
        np.testing.assert_allclose(tensor[0, :, 200, 60] * 255, (5, 87, 43))
        np.testing.assert_allclose(tensor[0, :, 200, 450] * 255, (19, 112, 71))
        self.assertEqual((transform.pad_x, transform.pad_y), (0, 160))
        with self.assertRaisesRegex(ValueError, '640'):
            prepare_legacy_pair(left[:100], right[:100], preprocessed=True)

    def test_prepare_center_crops_stitches_and_letterboxes_in_model_channel_order(self):
        left = np.full((4, 8, 3), 123, dtype=np.uint8)
        left[:, 2:4] = (255, 0, 0)
        left[:, 4:6] = 255
        right = np.full((8, 4, 3), 222, dtype=np.uint8)
        right[2:6] = 77

        stitched, tensor, transform = prepare_legacy_pair(left, right)

        self.assertEqual(stitched.shape, (320, 640, 3))
        self.assertFalse(np.shares_memory(stitched, left))
        self.assertFalse(np.shares_memory(stitched, right))
        self.assertTrue(np.all(stitched[:, :160] == (255, 0, 0)))
        self.assertTrue(np.all(stitched[:, 160:320] == 255))
        self.assertTrue(np.all(stitched[:, 320:] == 77))
        self.assertEqual(tensor.shape, (1, 3, 640, 640))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(tensor.flags.c_contiguous)
        self.assertTrue(np.all(tensor[:, :, :160] == np.float32(114 / 255)))
        self.assertTrue(np.all(tensor[:, :, 480:] == np.float32(114 / 255)))
        np.testing.assert_allclose(tensor[0, :, 300, 60] * 255, (0, 32, 29), atol=0.01)
        np.testing.assert_allclose(tensor[0, :, 300, 450] * 255, (0, 80, 77), atol=0.01)
        self.assertEqual(tensor[0, 0, 300, 159], 1.0)
        self.assertEqual((transform.original_width, transform.original_height), (640, 320))
        self.assertEqual((transform.pad_x, transform.pad_y), (0, 160))

    def test_prepare_rejects_invalid_frames(self):
        valid = np.zeros((4, 6, 3), dtype=np.uint8)
        for invalid in (None, valid[:, :0], valid[:, :, 0], valid.astype(np.float32)):
            with self.subTest(invalid=type(invalid)):
                with self.assertRaises(ValueError):
                    prepare_legacy_pair(valid, invalid)

    def test_legacy_decode_restores_box_and_retains_class_25(self):
        view = np.zeros((4, 6, 3), dtype=np.uint8)
        _, _, transform = prepare_legacy_pair(view, view)
        classes = load_class_names(CLASSES_FILE, expected_count=26)
        raw = np.zeros((1, 30, 8400), dtype=np.float32)
        raw[0, :4, 0] = (400, 210, 20, 10)
        raw[0, 4 + 25, 0] = 0.9

        with self.assertRaisesRegex(ValueError, "exactly 23"):
            decode_yolo_raw(raw, transform, classes)
        result = decode_yolo_raw(raw, transform, classes, 0.45, 0.50, expected_count=26)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].class_id, 25)
        self.assertEqual(result[0].class_name, "ban_qiu_ke")
        self.assertEqual(result[0].box, (390, 45, 410, 55))

    def test_legacy_class_file_rejects_duplicate_reserved_or_extra_names(self):
        names = ["class_%d" % index for index in range(26)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "classes.txt"
            path.write_text("\n".join(names), encoding="utf-8")
            self.assertEqual(load_class_names(path, expected_count=26), names)
            for bad_names, message in (
                (names[:-1] + [names[0]], "unique"),
                (names[:-1] + ["reserved_25"], "reserved"),
                (names + ["extra"], "26"),
            ):
                with self.subTest(message=message):
                    path.write_text("\n".join(bad_names), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, message):
                        load_class_names(path, expected_count=26)


class FakeBackend:
    def __init__(self, *outputs):
        self.outputs = iter(outputs)
        self.inputs = []
        self.close_count = 0

    def infer(self, tensor):
        self.inputs.append(tensor)
        output = next(self.outputs)
        if isinstance(output, Exception):
            raise output
        return output

    def close(self):
        self.close_count += 1


def _raw(*rows):
    output = np.zeros((1, 30, 8400), dtype=np.float32)
    for index, (box, class_id, score) in enumerate(rows):
        x1, y1, x2, y2 = box
        output[0, :4, index] = ((x1 + x2) / 2, (y1 + y2) / 2 + 160, x2 - x1, y2 - y1)
        output[0, 4 + class_id, index] = score
    return output


class LegacyRecognizerTests(unittest.TestCase):
    def setUp(self):
        self.config = yaml.safe_load((CLASSES_FILE.parents[1] / "legacy_vision.yaml").read_text(
            encoding="utf-8"
        ))
        self.classes = load_class_names(CLASSES_FILE, expected_count=26)
        self.left = np.full((320, 320, 3), (100, 0, 0), dtype=np.uint8)
        self.right = np.full((320, 320, 3), (100, 0, 80), dtype=np.uint8)
        self.left_box = ((40, 80, 140, 160), 7, 0.7)
        self.right_box = ((400, 80, 480, 160), 25, 0.9)
        self.empty = np.full((1, 1, 320, 320), -5, dtype=np.float32)
        self.ring = self.empty.copy()
        self.ring[0, 0, 90:150, 100:140] = 5
        self.ring[0, 0, 110:130, 115:125] = -5

    def make_recognizer(self, raw, *masks):
        self.detector = FakeBackend(raw)
        self.segmenter = FakeBackend(*masks) if masks else None
        return LegacyRecognizer(self.detector, self.segmenter, self.classes, self.config)

    def test_highest_confidence_wins_shape_and_right_overlay_preserves_the_hole(self):
        left_before, right_before = self.left.copy(), self.right.copy()
        source = np.hstack((left_before, right_before))
        recognizer = self.make_recognizer(_raw(self.left_box, self.right_box), self.empty, self.ring)
        result = recognizer.recognize_pair(self.left, self.right)
        self.assertEqual(result.detection.class_id, 25)
        self.assertEqual(result.mask_status, ("empty", "applied"))
        np.testing.assert_array_equal(result.preview_bgr[120, 440], source[120, 440])
        self.assertTrue(np.any(result.preview_bgr[100, 430] != source[100, 430]))
        np.testing.assert_array_equal(result.preview_bgr[100, 110], source[100, 110])
        np.testing.assert_array_equal(self.left, left_before)
        np.testing.assert_array_equal(self.right, right_before)
        self.assertEqual(len(self.detector.inputs), 1)
        self.assertEqual(self.detector.inputs[0].shape, (1, 3, 640, 640))
        self.assertEqual(self.segmenter.inputs[1].shape, (1, 3, 320, 320))
        # The supplied mask was trained on feature half-images, not raw RGB.
        np.testing.assert_allclose(self.segmenter.inputs[1], self.detector.inputs[0][:, :, 160:480, 320:640])
        self.assertGreaterEqual(result.elapsed_ms, 0)

    def test_no_detection_skips_masks_and_returns_an_unmarked_preview(self):
        recognizer = self.make_recognizer(_raw())
        result = recognizer.recognize_pair(self.left, self.right)
        self.assertIsNone(result.detection)
        self.assertIsNone(result.color)
        self.assertIsNone(result.error)
        self.assertEqual(result.mask_status, ("no_detection", "no_detection"))
        np.testing.assert_array_equal(result.preview_bgr, np.hstack((self.left, self.right)))

    def test_feature_image_runs_feature_mask_without_inventing_color(self):
        from unittest.mock import patch
        recognizer = self.make_recognizer(_raw(self.left_box, self.right_box), self.empty, self.ring)
        with patch('legacy_vision.classify_hsv', side_effect=AssertionError('false feature color')):
            result = recognizer.recognize_pair(self.left, self.right, preprocessed=True)
        self.assertIsNone(result.error)
        self.assertEqual(result.detection.class_id, 25)
        self.assertIsNone(result.color)
        self.assertEqual(result.mask_status, ('empty', 'applied'))
        self.assertEqual(len(self.segmenter.inputs), 2)
        np.testing.assert_allclose(self.segmenter.inputs[1][0, :, 200, 60] * 255, (80, 0, 100))
        np.testing.assert_allclose(self.detector.inputs[0][0, :, 200, 60] * 255, (0, 0, 100))

    def test_mask_tightens_box_and_excludes_background_and_hole_from_color(self):
        # Purple background fills most of the detection; only the ring is blue.
        self.left[:] = (100, 0, 80)
        mask = np.full((1, 1, 320, 320), -5, dtype=np.float32)
        mask[0, 0, 90:150, 60:120] = 5
        mask[0, 0, 105:135, 75:105] = -5
        foreground = mask[0, 0] > 0
        self.left[foreground] = (100, 0, 0)
        result = self.make_recognizer(_raw(self.left_box), mask).recognize_pair(self.left, self.right)
        self.assertEqual(result.color.name, '蓝色')
        self.assertEqual(result.color.valid_pixels, 2700)
        self.assertEqual(result.color_sources, ('mask', 'none'))
        self.assertEqual(result.surface_boxes[0], (60, 90, 120, 150))
        self.assertEqual(result.hole_counts, (1, None))
        # A binary surface model supplies geometry, never a replacement class.
        self.assertEqual(result.detection.class_id, self.left_box[1])

    def test_mask_uses_only_the_component_associated_with_the_detection(self):
        mask = np.full((1, 1, 320, 320), -5, dtype=np.float32)
        mask[0, 0, 90:150, 60:120] = 5
        mask[0, 0, 200:300, 200:300] = 5
        result = self.make_recognizer(_raw(self.left_box), mask).recognize_pair(self.left, self.right)
        np.testing.assert_array_equal(result.preview_bgr[250, 250], self.left[250, 250])
        self.assertEqual(result.surface_boxes[0], (60, 90, 120, 150))

    def test_explicit_raw_rgb_mask_release_keeps_its_own_input_contract(self):
        self.config['surface_mask'] = {'input_mode': 'raw_rgb'}
        recognizer = self.make_recognizer(_raw(self.right_box), self.ring)
        recognizer.recognize_pair(self.left, self.right)
        np.testing.assert_allclose(self.segmenter.inputs[0][0, :, 120, 120], (80 / 255, 0, 100 / 255))

    def test_failed_mask_has_visible_fallback_color_source(self):
        result = self.make_recognizer(_raw(self.left_box), RuntimeError('bad mask')).recognize_pair(self.left, self.right)
        self.assertEqual(result.mask_status, ('error', 'no_detection'))
        self.assertEqual(result.color_sources, ('box', 'none'))
        self.assertEqual(result.hole_counts, (None, None))

    def test_unknown_mask_input_and_incompatible_feature_image_do_not_run_mask(self):
        for mode in ('unknown', 'raw_rgb'):
            with self.subTest(mode=mode):
                self.config['surface_mask'] = {'input_mode': mode}
                recognizer = self.make_recognizer(_raw(self.left_box), self.ring)
                result = recognizer.recognize_pair(self.left, self.right, preprocessed=True)
                self.assertEqual(result.mask_status[0], 'input_mismatch')
                self.assertEqual(len(self.segmenter.inputs), 0)

    def test_overfull_mask_is_rejected_instead_of_coloring_background(self):
        full = np.ones((1, 1, 320, 320), dtype=np.float32) * 5
        result = self.make_recognizer(_raw(self.left_box), full).recognize_pair(self.left, self.right)
        self.assertEqual(result.mask_status[0], 'overfull')
        self.assertEqual(result.color_sources[0], 'box')
        self.assertIsNone(result.hole_counts[0])

    def test_multiple_left_candidates_skip_only_the_left_overlay(self):
        extra_left = ((160, 80, 200, 160), 4, 0.6)
        recognizer = self.make_recognizer(_raw(self.left_box, extra_left, self.right_box), self.ring)
        result = recognizer.recognize_pair(self.left, self.right)
        self.assertEqual(result.mask_status, ("ambiguous", "applied"))
        self.assertEqual(len(self.segmenter.inputs), 1)

    def test_segmentation_exception_and_nonintersecting_mask_preserve_shape(self):
        far_right_box = ((500, 80, 580, 160), 25, 0.9)
        recognizer = self.make_recognizer(
            _raw(self.left_box, far_right_box), RuntimeError("segmentation failed"), self.ring
        )
        result = recognizer.recognize_pair(self.left, self.right)
        self.assertEqual(result.mask_status, ("error", "unmatched"))
        self.assertEqual(result.detection.class_id, 25)
        self.assertIsNone(result.error)
        np.testing.assert_array_equal(result.preview_bgr[100, 430], self.right[100, 110])

    def test_detection_failure_returns_error_without_running_segmentation(self):
        recognizer = self.make_recognizer(RuntimeError("detector unavailable"), self.ring)
        result = recognizer.recognize_pair(self.left, self.right)
        self.assertIn("detector unavailable", result.error)
        self.assertIsNone(result.detection)
        self.assertIsNone(result.color)
        self.assertEqual(len(self.segmenter.inputs), 0)

    def test_cross_seam_tie_assigns_left_and_color_prefers_a_known_view(self):
        self.right[:] = (10, 0, 10)  # Saturated but below every hue range's value floor.
        tied_box = ((280, 80, 360, 160), 7, 0.7)
        recognizer = self.make_recognizer(_raw(tied_box, self.right_box), self.empty, self.empty)
        result = recognizer.recognize_pair(self.left, self.right)
        self.assertEqual(result.color.name, "\u84dd\u8272")
        self.assertEqual(result.color.valid_pixels, 40 * 80)
        self.assertEqual(result.mask_status, ("empty", "empty"))

    def test_color_ranks_hsv_confidence_before_detection_confidence(self):
        self.right[80:160, 80:100] = (10, 0, 10)
        raw = _raw(self.left_box, self.right_box)
        result = self.make_recognizer(raw).recognize_pair(self.left, self.right)
        self.assertEqual(result.color.name, "\u84dd\u8272")
        self.assertEqual(result.mask_status, ("disabled", "disabled"))
        self.right[:] = (100, 0, 80)
        result = self.make_recognizer(raw).recognize_pair(self.left, self.right)
        self.assertEqual(result.color.name, "\u7d2b\u8272")

    def test_close_closes_each_backend_once(self):
        recognizer = self.make_recognizer(_raw(), self.empty)
        recognizer.close()
        recognizer.close()
        self.assertEqual(self.detector.close_count, 1)
        self.assertEqual(self.segmenter.close_count, 1)
