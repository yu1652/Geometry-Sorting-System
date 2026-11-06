import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from vision_worker import load_model_set


class SurfaceLoadingTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        (self.root / 'models').mkdir()
        (self.root / 'models/legacy_classes.txt').write_text('\n'.join('c%d' % i for i in range(26)))
        (self.root / 'legacy_vision.yaml').write_text(yaml.safe_dump({
            'surface_mask': {'enabled': True, 'input_mode': 'legacy_features'}}))
        self.calls = []

    def backend(self, path, model_kind):
        self.calls.append((Path(path).name, model_kind))
        return SimpleNamespace(close=lambda: None)

    def load(self, releases=None):
        with patch('vision_worker.ROOT', self.root), patch('trt_backend.create_backend', self.backend):
            recognizer = load_model_set(releases or {})
        self.addCleanup(recognizer.close)
        return recognizer

    def test_supplied_local_mask_engine_is_loaded_by_default(self):
        (self.root / 'models/surface_mask_fp16.engine').write_bytes(b'fixture')
        recognizer = self.load()
        self.assertEqual(self.calls, [('shape_legacy_fp16.engine', 'yolo_legacy'),
                                     ('surface_mask_fp16.engine', 'mask')])
        self.assertIsNotNone(recognizer._segmenter)
        self.assertEqual(recognizer._config['surface_mask']['input_mode'], 'legacy_features')

    def test_missing_engine_is_explicit_and_preserves_detector(self):
        recognizer = self.load()
        self.assertIsNone(recognizer._segmenter)
        self.assertEqual(recognizer._config['surface_mask']['unavailable_status'], 'missing_model')

    def test_release_preprocessing_overrides_local_default_without_changing_detector(self):
        for preprocess, expected in [('surface_rgb320_v1', 'raw_rgb'),
                                     ('surface_features320_v1', 'legacy_features')]:
            with self.subTest(preprocess=preprocess):
                release = SimpleNamespace(engine_path=self.root / 'released.engine',
                                          manifest={'preprocess': preprocess})
                recognizer = self.load({'surface_mask': release})
                self.assertEqual(recognizer._config['surface_mask']['input_mode'], expected)
                self.assertEqual(self.calls[-1], ('released.engine', 'mask'))
