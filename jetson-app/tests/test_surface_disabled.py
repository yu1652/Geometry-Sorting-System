from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import yaml

import butu_20261110 as butu
from side_model_store import LiveSideProfile
from test_live_side_model import NAMES

ROOT = Path(__file__).resolve().parents[1]


class Head:
    def __init__(self, count):
        self.count = count
        self.inputs = []
        self.closed = False

    def infer(self, tensor):
        self.inputs.append(tensor.copy())
        raw = np.zeros((1, 4 + self.count, 2100), np.float32)
        raw[0, :4, 0] = [160, 160, 80, 80]
        raw[0, 4, 0] = .95
        return raw

    def close(self):
        self.closed = True


class DisabledSurfaceTests(unittest.TestCase):
    def make_root(self, folder):
        root = Path(folder)
        (root / 'models').mkdir()
        # An existing engine and enabled YAML must not accidentally reactivate it.
        (root / 'models' / 'surface_mask_fp16.engine').write_bytes(b'mask exists')
        config = yaml.safe_load((ROOT / 'legacy_vision.yaml').read_text(encoding='utf8'))
        config['surface_mask']['enabled'] = True
        (root / 'legacy_vision.yaml').write_text(yaml.safe_dump(config), encoding='utf8')
        return root

    def assert_shape_and_color_still_work(self, recognizer):
        image = np.full((320, 320, 3), (0, 0, 180), np.uint8)
        result = recognizer.recognize_pair(image, image)
        self.assertIsNone(result.error)
        self.assertIsNotNone(result.detection)
        self.assertEqual(result.mask_status, ('disabled', 'disabled'))
        self.assertEqual(result.color_sources, ('box', 'box'))
        self.assertEqual(result.color.name, '红色')
        self.assertIsNone(recognizer._segmenter)
        # Detector still receives features, not raw red RGB pixels.
        self.assertLess(float(recognizer._detector.inputs[0][0, 0].mean()), .05)

    def test_old_24_class_profile_never_loads_surface_engine(self):
        with tempfile.TemporaryDirectory() as d:
            root = self.make_root(d)
            calls = []
            def backend(path, model_kind, **kwargs):
                calls.append(model_kind)
                if model_kind == 'mask':
                    raise AssertionError('surface mask must not load')
                return Head(24)
            with patch.object(butu, 'ROOT', root), patch.object(butu, 'MODEL_DIR', root / 'models'), patch('trt_backend.create_backend', backend):
                recognizer = butu.make_recognizer()
            try:
                self.assertEqual(calls, ['yolo_butu24'])
                self.assert_shape_and_color_still_work(recognizer)
            finally:
                recognizer.close()

    def test_imported_ten_class_profile_never_loads_surface_engine(self):
        with tempfile.TemporaryDirectory() as d:
            root = self.make_root(d)
            record = {'class_names': NAMES, 'engine_path': str(root / 'shape.engine'), 'preprocess': 'legacy_features'}
            profile = LiveSideProfile(root, SimpleNamespace(MODEL_DIR=root / 'models'), record)
            calls = []
            def backend(path, model_kind, **kwargs):
                calls.append(model_kind)
                if model_kind == 'mask':
                    raise AssertionError('surface mask must not load')
                return Head(10)
            with patch('side_model_store.create_backend', backend):
                recognizer = profile.make_recognizer()
            try:
                self.assertEqual(calls, ['yolo_side'])
                self.assert_shape_and_color_still_work(recognizer)
            finally:
                recognizer.close()
