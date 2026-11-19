import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from legacy_vision import LegacyRecognizer
from feature_preprocess import feature_bgr

NAMES = ('ban_yuan_zhu', 'chang_fang_ti', 'liu_leng_zhu', 'san_leng_zhu', 'san_leng_zhui',
         'si_leng_zhui', 'wu_leng_zhu', 'yuan_zhu', 'yuan_zhui', 'zheng_fang_ti')
ROOT = Path(__file__).resolve().parents[1]


class Detector:
    def __init__(self, bad=False):
        self.inputs = []
        self.bad = bad
        self.closed = False

    def infer(self, tensor):
        self.inputs.append(tensor.copy())
        out = np.zeros((1, 14, 2100), np.float32)
        out[0, :4, 0] = [160, 160, 80, 80]
        out[0, 13, 0] = np.nan if self.bad else .95
        return out

    def close(self):
        self.closed = True


def make_bundle(root):
    bundle = root / 'models' / 'imports' / ('a' * 32)
    bundle.mkdir(parents=True)
    (root / 'data').mkdir()
    (root / 'data' / 'active_model.json').write_text(json.dumps({'bundle': bundle.name}))
    (bundle / 'model_fp16.engine').write_bytes(b'engine')
    (bundle / 'model.onnx').write_bytes(b'onnx')
    (bundle / 'manifest.json').write_text(json.dumps({
        'names': NAMES, 'input_size': 320, 'output_shape': [1, 14, 2100],
        'source_name': 'best.onnx', 'sha256': hashlib.sha256(b'onnx').hexdigest()}))
    return bundle


class SideModelTests(unittest.TestCase):
    def test_tensor_contract_and_old_numpy_mapping_are_checked_without_np_bool(self):
        from test_trt_backend import FakeEngine, FakeTrt, FakeCuda
        from trt_backend import TensorRTBackend, EngineContractError
        events = []
        engine = FakeEngine(events, output_shape=(1, 14, 2100))
        trt = FakeTrt(engine)
        trt.float16, trt.float32 = 'fp16', 'fp32'
        trt.nptype = lambda dtype: (_ for _ in ()).throw(AttributeError('np.bool removed'))
        backend = TensorRTBackend(b'engine', 'yolo_side', trt, FakeCuda(events), side_class_count=10)
        try:
            self.assertEqual(backend.infer(np.zeros((1, 3, 320, 320))).shape, (1, 14, 2100))
        finally:
            backend.close()
        wrong = FakeEngine([], output_shape=(1, 28, 2100))
        with self.assertRaises(EngineContractError):
            TensorRTBackend(b'engine', 'yolo_side', FakeTrt(wrong), FakeCuda([]), side_class_count=10)

    def test_raw_rgb_mode_does_not_feed_features_to_detector(self):
        detector = Detector()
        config = yaml.safe_load((ROOT / 'legacy_vision.yaml').read_text(encoding='utf8'))
        recognizer = LegacyRecognizer(detector, None, NAMES, config, expected_count=10,
                                      per_view_features=True, per_view_input_mode='raw_rgb')
        image = np.full((320, 320, 3), (10, 80, 180), np.uint8)
        result = recognizer.recognize_pair(image, image)
        self.assertIsNone(result.error)
        np.testing.assert_allclose(detector.inputs[0][0, :, 0, 0] * 255, [180, 80, 10], atol=1)
        self.assertIsNotNone(recognizer.recognize_pair(image, image, preprocessed=True).error)

    def test_model_manager_confirmation_requires_restart_before_device_reconnect(self):
        from side_model_store import SideModelStore
        import live_console
        application = live_console.QApplication.instance() or live_console.QApplication([])
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            make_bundle(root / 'offline')
            store = SideModelStore(root / 'live')
            with patch.object(live_console, 'ROOT', root / 'live'):
                window = live_console.LiveWindow(data_dir=root / 'live' / 'data', butu_24=True)
                try:
                    self.assertEqual(len(window.names), 24)
                    with patch('side_model_store.create_backend', return_value=Detector()):
                        candidate = store.prepare_existing(root / 'offline')
                    store.activate(candidate)
                    window._model_process = SimpleNamespace(poll=lambda: 0)
                    self.assertTrue(window._models_open())
                    window.connect_cameras()
                    window.connect_serial()
                    self.assertTrue(window._model_restart_required)
                    self.assertIsNone(window.vision)
                    self.assertIsNone(window.serial_port)
                finally:
                    window.close()

    def test_ten_class_model_uses_existing_feature_pipeline_and_original_color(self):
        detector = Detector()
        config = yaml.safe_load((ROOT / 'legacy_vision.yaml').read_text(encoding='utf8'))
        recognizer = LegacyRecognizer(detector, None, NAMES, config,
                                      expected_count=10, per_view_features=True)
        left = np.full((320, 320, 3), (0, 0, 180), np.uint8)
        right = left.copy()
        result = recognizer.recognize_pair(left, right)
        self.assertIsNone(result.error)
        self.assertEqual(result.detection.class_name, 'zheng_fang_ti')
        self.assertEqual(result.color.name, '红色')
        expected = feature_bgr(np.hstack((left, right)))
        self.assertEqual(len(detector.inputs), 2)
        np.testing.assert_allclose(detector.inputs[0][0] * 255,
                                   expected[:, :320, ::-1].transpose(2, 0, 1), atol=1)

    def test_reuses_engine_without_rebuilding_and_only_activates_after_confirmation(self):
        from side_model_store import SideModelStore
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            make_bundle(root / 'offline')
            store = SideModelStore(root / 'live')
            backend = Detector()
            with patch('side_model_store.create_backend', return_value=backend), patch('side_model_store.subprocess.run') as build:
                candidate = store.prepare_existing(root / 'offline')
            build.assert_not_called()
            self.assertTrue(backend.closed)
            self.assertFalse(store.active_path.exists())
            store.activate(candidate)
            active = store.read_active()
            self.assertEqual(active['class_names'], list(NAMES))
            self.assertEqual(active['preprocess'], 'legacy_features')
            self.assertEqual(active['engine_sha256'], hashlib.sha256(b'engine').hexdigest())

    def test_failed_trial_keeps_previous_pointer(self):
        from side_model_store import SideModelStore
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            make_bundle(root / 'offline')
            store = SideModelStore(root / 'live')
            store.active_path.parent.mkdir(parents=True)
            store.active_path.write_text('{"existing":"pointer"}')
            before = store.active_path.read_bytes()
            with patch('side_model_store.create_backend', return_value=Detector(bad=True)):
                with self.assertRaises(ValueError):
                    store.prepare_existing(root / 'offline')
            self.assertEqual(store.active_path.read_bytes(), before)

    def test_full_live_profile_reads_ten_classes_despite_butu24_flag(self):
        from side_model_store import SideModelStore
        import live_console
        application = live_console.QApplication.instance() or live_console.QApplication([])
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            make_bundle(root / 'offline')
            store = SideModelStore(root / 'live')
            with patch('side_model_store.create_backend', return_value=Detector()):
                candidate = store.prepare_existing(root / 'offline')
            store.activate(candidate)
            with patch.object(live_console, 'ROOT', root / 'live'):
                window = live_console.LiveWindow(data_dir=root / 'live' / 'data', butu_24=True)
            try:
                self.assertEqual(tuple(window.names), NAMES)
                self.assertEqual(window._shape_name('chang_fang_ti'), '长方体')
                self.assertIn('10', window.windowTitle())
                self.assertFalse(window.machine.snapshot().origin_confirmed)
                self.assertIsNone(window.serial_port)
                self.assertIsNone(window.vision)
                self.assertIn('live_side_', window.auto_bins_path.name)
                self.assertEqual(window.pages.count(), 2)
            finally:
                window.close()


if __name__ == '__main__':
    unittest.main()
