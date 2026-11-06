import io
import json
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime_diagnostics import RuntimeDiagnostics


class RuntimeDiagnosticsTests(unittest.TestCase):
    def test_sample_preserves_images_arrays_and_metadata(self):
        image = np.arange(72, dtype=np.uint8).reshape(4, 6, 3)
        raw = np.array([[0.2, 0.9]], dtype=np.float32)
        original = image.copy()
        result = SimpleNamespace(
            detection=SimpleNamespace(class_id=np.int64(2), class_name='圆形', confidence=np.float32(.9), box=(1, 2, 3, 4)),
            color=SimpleNamespace(name='red', confidence=.8, valid_pixels=12),
            mask_status=('ok', 'empty'), hole_counts=(1, None), color_sources=('mask', 'none'),
            elapsed_ms=3.4, error=None, preview_bgr=image,
            diagnostics={'pair_bgr': image, 'features_bgr': image, 'raw_output': raw,
                         'captures': {'left': image}})
        with tempfile.TemporaryDirectory() as folder:
            recorder = RuntimeDiagnostics(folder)
            path = recorder.save_sample(result, {'preprocessed': True, 'reason': 'empty'})
            with zipfile.ZipFile(path) as bundle:
                metadata = json.loads(bundle.read('metadata.json'))
                self.assertEqual(metadata['detection']['class_name'], '圆形')
                self.assertEqual(metadata['detection']['box'], [1, 2, 3, 4])
                self.assertEqual(metadata['color']['valid_pixels'], 12)
                self.assertEqual(metadata['mask_status'], ['ok', 'empty'])
                self.assertEqual(metadata['hole_counts'], [1, None])
                self.assertEqual(metadata['color_sources'], ['mask', 'none'])
                self.assertEqual(metadata['elapsed_ms'], 3.4)
                self.assertIsNone(metadata['error'])
                self.assertTrue(metadata['context']['preprocessed'])
                for name in ('pair_bgr.png', 'features_bgr.png', 'preview_bgr.png', 'captures/left.png'):
                    decoded = cv2.imdecode(np.frombuffer(bundle.read(name), dtype=np.uint8), cv2.IMREAD_COLOR)
                    np.testing.assert_array_equal(decoded, original)
                np.testing.assert_array_equal(np.load(io.BytesIO(bundle.read('raw_output.npy')), allow_pickle=False), raw)
            np.testing.assert_array_equal(image, original)
            np.testing.assert_array_equal(raw, [[np.float32(.2), np.float32(.9)]])
            path.unlink()  # all handles must be closed on Windows

    def test_missing_optional_fields_still_records_context(self):
        with tempfile.TemporaryDirectory() as folder:
            path = RuntimeDiagnostics(folder).save_sample(SimpleNamespace(), {'reason': 'empty'})
            with zipfile.ZipFile(path) as bundle:
                metadata = json.loads(bundle.read('metadata.json'))
                self.assertIsNone(metadata['detection'])
                self.assertIsNone(metadata['color'])
                self.assertEqual(metadata['context'], {'reason': 'empty'})
                self.assertEqual(bundle.namelist(), ['metadata.json'])

    def test_log_unicode_multiline_monotonic_and_bounded_rotation(self):
        with tempfile.TemporaryDirectory() as folder:
            recorder = RuntimeDiagnostics(folder, log_max_bytes=240)
            recorder.log('开始\n自动扫描')
            log = Path(folder) / 'live_runtime.log'
            value = log.read_text(encoding='utf-8')
            self.assertIn('开始', value)
            self.assertIn('自动扫描', value)
            self.assertRegex(value, r'\d{4}-\d{2}-\d{2}T.*monotonic=\d+\.\d+')
            for index in range(15):
                recorder.log('iteration %d %s' % (index, 'x' * 100))
            self.assertLessEqual(log.stat().st_size, 240)
            self.assertEqual(len(list(Path(folder).glob('live_runtime.log*'))), 4)
            self.assertIn('iteration 14', log.read_text(encoding='utf-8'))
            for file in Path(folder).glob('live_runtime.log*'):
                file.rename(file.with_suffix(file.suffix + '.closed'))

    def test_log_write_error_is_propagated(self):
        with tempfile.TemporaryDirectory() as folder:
            recorder = RuntimeDiagnostics(folder)
            (Path(folder) / 'live_runtime.log').mkdir()
            with self.assertRaises(OSError):
                recorder.log('test')

    def test_samples_and_exports_have_unique_names_and_keep_recent(self):
        with tempfile.TemporaryDirectory() as folder:
            recorder = RuntimeDiagnostics(folder, max_samples=2)
            paths = [recorder.save_sample(SimpleNamespace(), {'index': index}) for index in range(4)]
            self.assertEqual(len(set(paths)), 4)
            self.assertFalse(paths[1].exists())
            self.assertTrue(paths[-1].exists())
            exports = [recorder.export() for _ in range(5)]
            self.assertEqual(len(set(exports)), 5)
            self.assertFalse(exports[1].exists())
            self.assertTrue(exports[-1].exists())
            self.assertEqual(len(list((Path(folder) / 'diagnostic_exports').glob('*.zip'))), 3)

    def test_export_only_collects_explicit_safe_files_and_recorded_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            recorder = RuntimeDiagnostics(root)
            recorder.log('one')
            recorder.log('two')
            sample = recorder.save_sample(SimpleNamespace(), {})
            config = root / 'config.json'
            config.write_text('{"camera": 0}', encoding='utf-8')
            model = root / 'model.engine'
            model.write_bytes(b'model')
            secret = root / 'secrets.json'
            secret.write_text('{"password": "private"}', encoding='utf-8')
            unrequested = root / 'other.json'
            unrequested.write_text('{}', encoding='utf-8')
            path = recorder.export((config, model, secret, root), metadata={'device': 'xavier'})
            with zipfile.ZipFile(path) as bundle:
                self.assertIn('live_runtime.log', bundle.namelist())
                self.assertIn('diagnostic_samples/' + sample.name, bundle.namelist())
                self.assertEqual(bundle.read('extra_files/config.json'), config.read_bytes())
                self.assertEqual(json.loads(bundle.read('metadata.json')), {'device': 'xavier'})
                self.assertFalse(any('model' in name or 'secret' in name or 'other.json' in name for name in bundle.namelist()))
            path.unlink()

    def test_failed_sample_and_export_remove_temporary_archives(self):
        with tempfile.TemporaryDirectory() as folder:
            recorder = RuntimeDiagnostics(folder)
            image = np.zeros((2, 2, 3), np.uint8)
            with patch('runtime_diagnostics.cv2.imencode', return_value=(False, None)):
                with self.assertRaises(ValueError):
                    recorder.save_sample(SimpleNamespace(preview_bgr=image), {})
            self.assertEqual(list((Path(folder) / 'diagnostic_samples').iterdir()), [])
            with patch('runtime_diagnostics.os.replace', side_effect=OSError('disk error')):
                with self.assertRaises(OSError):
                    recorder.export()
            self.assertEqual(list((Path(folder) / 'diagnostic_exports').iterdir()), [])


if __name__ == '__main__':
    unittest.main()
