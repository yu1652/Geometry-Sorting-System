import unittest

import numpy as np
import yaml

from butu_20261110 import CLASS_NAMES, MODEL_DIR, ROOT, SHAPES
from feature_preprocess import feature_bgr
from legacy_vision import LegacyRecognizer


class Detector:
    def __init__(self):
        self.inputs = []

    def infer(self, tensor):
        self.inputs.append(tensor.copy())
        output = np.zeros((1, 28, 2100), dtype=np.float32)
        output[0, :4, 0] = (100, 120, 40, 60)
        output[0, 4 + len(self.inputs) - 1, 0] = .9
        return output

    def close(self):
        pass


class ButuTests(unittest.TestCase):
    def test_class_order_and_per_view_feature_inputs(self):
        self.assertEqual(CLASS_NAMES, tuple((MODEL_DIR / 'classes.txt').read_text(encoding='utf8').splitlines()))
        self.assertEqual(len(CLASS_NAMES), 24)
        self.assertEqual(set(CLASS_NAMES), set(SHAPES))
        detector = Detector()
        config = yaml.safe_load((ROOT / 'legacy_vision.yaml').read_text(encoding='utf8'))
        recognizer = LegacyRecognizer(detector, None, CLASS_NAMES, config,
                                      expected_count=24, per_view_features=True)
        left = np.full((320, 320, 3), (40, 90, 130), dtype=np.uint8)
        right = np.full((320, 320, 3), (80, 140, 210), dtype=np.uint8)
        result = recognizer.recognize_pair(left, right)
        self.assertIsNone(result.error)
        self.assertEqual(len(detector.inputs), 2)
        self.assertEqual(result.diagnostics['raw_output'].shape, (2, 1, 28, 2100))
        expected = feature_bgr(np.hstack((left, right)))
        for side in range(2):
            np.testing.assert_allclose(detector.inputs[side][0] * 255,
                                       expected[:, side * 320:(side + 1) * 320, ::-1].transpose(2, 0, 1), atol=1)
        self.assertEqual(result.detection.box, (80.0, 90.0, 120.0, 150.0))
        self.assertEqual(result.detection.class_name, CLASS_NAMES[0])
        self.assertEqual(result.mask_status, ('disabled', 'disabled'))

    def test_color_cannot_come_from_a_disagreeing_shape_view(self):
        detector = Detector()
        config = yaml.safe_load((ROOT / 'legacy_vision.yaml').read_text(encoding='utf8'))
        recognizer = LegacyRecognizer(detector, None, CLASS_NAMES, config,
                                      expected_count=24, per_view_features=True)
        left = np.full((320, 320, 3), (95, 170, 55), dtype=np.uint8)
        right = np.full((320, 320, 3), (0, 0, 180), dtype=np.uint8)
        result = recognizer.recognize_pair(left, right)
        self.assertEqual(result.detection.class_id, 0)
        self.assertEqual(result.color_sources, ('box', 'none'))
        self.assertNotEqual(result.color.name, '红色')

    def test_preprocessed_input_is_not_processed_twice(self):
        detector = Detector()
        recognizer = LegacyRecognizer(detector, None, CLASS_NAMES,
                                      {'surface_mask': {'input_mode': 'legacy_features'}},
                                      expected_count=24, per_view_features=True)
        left = np.full((320, 320, 3), (10, 20, 30), dtype=np.uint8)
        right = np.full((320, 320, 3), (40, 50, 60), dtype=np.uint8)
        recognizer.recognize_pair(left, right, preprocessed=True)
        np.testing.assert_allclose(detector.inputs[0][0, :, 0, 0] * 255, (30, 20, 10), atol=1)
        np.testing.assert_allclose(detector.inputs[1][0, :, 0, 0] * 255, (60, 50, 40), atol=1)


if __name__ == '__main__':
    unittest.main()
