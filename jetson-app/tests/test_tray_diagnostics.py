"""High-frequency tray archives cannot evict recognition evidence."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from zipfile import ZipFile

import cv2
import numpy as np
from runtime_diagnostics import RuntimeDiagnostics


class TrayDiagnosticsTests(unittest.TestCase):
    def test_separate_retention_and_export_include_both_groups_and_exact_top(self):
        with tempfile.TemporaryDirectory() as folder:
            recorder = RuntimeDiagnostics(folder, max_samples=2)
            recognition = recorder.save_sample(SimpleNamespace(), {'kind': 'sort'})
            frame = np.full((480, 640, 3), 29, np.uint8)
            result = SimpleNamespace(diagnostics={'captures': {'top': frame},
                'tray': {'locator': {'selection_status': 'selected'}, 'detector_kind': 'foreground'},
                'foreground_mask': np.full((20, 30), 255, np.uint8)})
            paths = [recorder.save_sample(result, {'token': token}, group='tray') for token in range(4)]
            self.assertTrue(recognition.exists())
            self.assertFalse(paths[0].exists())
            self.assertEqual(paths[-1].parent.name, 'diagnostic_tray_samples')
            self.assertTrue(paths[-1].name.startswith('tray_'))
            with ZipFile(str(paths[-1])) as archive:
                actual = cv2.imdecode(np.frombuffer(archive.read('captures/top.png'), np.uint8), cv2.IMREAD_COLOR)
                np.testing.assert_array_equal(actual, frame)
                self.assertEqual(json.loads(archive.read('tray.json'))['detector_kind'], 'foreground')
                self.assertIn('foreground_mask.png', archive.namelist())
            with ZipFile(str(recorder.export())) as archive:
                self.assertIn('diagnostic_samples/' + recognition.name, archive.namelist())
                self.assertIn('diagnostic_tray_samples/' + paths[-1].name, archive.namelist())

    def test_invalid_group_cannot_create_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            recorder = RuntimeDiagnostics(folder)
            with self.assertRaises(ValueError):
                recorder.save_sample(SimpleNamespace(), {}, group='../elsewhere')
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_only_explicit_named_tray_reference_is_exportable(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            reference = root / 'tray_background.npz'
            reference.write_bytes(b'reference')
            weights = root / 'weights.npz'
            weights.write_bytes(b'weights')
            with ZipFile(str(RuntimeDiagnostics(root).export((reference, weights)))) as archive:
                self.assertEqual(archive.read('extra_files/tray_background.npz'), b'reference')
                self.assertNotIn('extra_files/weights.npz', archive.namelist())


if __name__ == '__main__':
    unittest.main()
