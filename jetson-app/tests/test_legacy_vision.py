import tempfile
import unittest
from pathlib import Path

import numpy as np

from legacy_vision import prepare_legacy_pair
from vision import decode_yolo_raw, load_class_names


CLASSES_FILE = Path(__file__).resolve().parents[1] / "models" / "legacy_classes.txt"


class LegacyVisionTests(unittest.TestCase):
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
