import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from app import QApplication, QPushButton
from live_console import LiveWindow

APP = QApplication.instance() or QApplication([])


class NoPresenceChecksTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.window = LiveWindow(data_dir=self.folder.name, linux_20261107=True)
        self.window.timer.stop()
        self.addCleanup(self.window.close)
        self.addCleanup(setattr, self.window, 'vision', None)
        confirm = patch.object(self.window, '_ask_origin_unchanged', return_value=True, create=True)
        confirm.start()
        self.addCleanup(confirm.stop)
        self.sent = []
        machine = self.window.machine
        machine._send = self.sent.append
        machine.begin_handshake()
        for line in ('PONG', 'VERSION,ESP32_SORTER_XY_PUSHER,0.1.0', 'STATUS,MCU,WAIT_START,0,0,77'):
            machine.handle_mcu(line)
        self.assertTrue(machine.snapshot().origin_confirmed)
        self.window.save_calibration(self.window.calibration_rows)
        self.window.calibration_confirmed.setChecked(True)
        self.window._models_warm = True
        self.window._frames_at = time.monotonic()
        self.window.vision = SimpleNamespace(recognize=lambda *a, **kw: True, locate=lambda *a: True,
            capture_empty=lambda *a: self.fail('Empty-reference capture must not be requested'))
        self.sent.clear()

    def result(self, target=True):
        return SimpleNamespace(error=None,
            detection=SimpleNamespace(class_name=self.window.names[0], confidence=.9) if target else None,
            color=SimpleNamespace(name='蓝色', confidence=.9, valid_pixels=120) if target else None,
            elapsed_ms=20., preview_bgr=np.zeros((320, 640, 3), np.uint8))

    def reach_scan(self):
        machine = self.window.machine
        machine.start()
        machine.handle_mcu('STATUS,HOST_READY')
        machine.handle_mcu('STATUS,START')
        machine.handle_mcu('PUSH?')
        machine.provide_target(80, 80)
        machine.handle_mcu('SCAN')

    def scan(self, result):
        self.window._scan_next_at = 0
        self.window._dispatch_vision()
        self.window.on_recognized(self.window._job[0], result)

    def test_start_requires_no_empty_reference_or_capture(self):
        reference = Path(self.folder.name) / 'recognition_empty.npz'
        reference.write_bytes(b'old-reference-is-never-read-or-replaced')
        with patch('scene_presence.EmptySceneReference.assess', side_effect=AssertionError('no background comparison')):
            self.assertTrue(self.window.start())
        self.assertIn('HOST_READY', self.sent)
        self.assertIsNone(self.window._job)
        self.assertEqual(reference.read_bytes(), b'old-reference-is-never-read-or-replaced')

    def test_manual_recognition_reports_detector_without_background_judgment(self):
        with patch('scene_presence.EmptySceneReference.assess', side_effect=AssertionError('no background comparison')):
            self.window._job = (19, self.window.machine.epoch, 'manual')
            self.window.on_recognized(19, self.result(False))
        self.assertIn('未识别到', self.window.result_label.text())
        self.assertFalse(any(word in self.window.result_label.text() for word in ('空区', '为空')))

    def test_no_target_discards_after_bounded_scans_without_sort_or_count(self):
        self.reach_scan()
        with patch('scene_presence.EmptySceneReference.assess', side_effect=AssertionError('no background comparison')):
            for _ in range(3):
                self.scan(self.result(False))
        self.assertEqual(self.window.machine.snapshot().state, 'reject_opening')
        self.assertIn('SERVO2,OPEN', self.sent)
        self.assertNotIn('STOP', self.sent)
        # No next push is permitted until the floor has actually replied closed.
        self.assertNotIn('PUSH,WAIT', self.sent)
        self.assertFalse(any(line.startswith('SORT,') for line in self.sent))
        self.assertEqual(self.window.auto_bins.total, 0)
        self.assertEqual(self.window.bindings, {})

    def test_done_counts_after_valid_consensus_without_drop_verification(self):
        self.reach_scan()
        self.scan(self.result())
        self.assertFalse(any(line.startswith('SORT,') for line in self.sent))
        self.scan(self.result())
        self.assertIn('SORT,1', self.sent)
        self.assertEqual(self.window.auto_bins.total, 0)
        self.assertTrue(self.window.machine.handle_mcu('DONE'))
        self.assertEqual(self.window.machine.snapshot().completed_count, 1)
        self.assertEqual(self.window.machine.snapshot().state, 'running')
        self.window.record_completion()
        self.assertEqual(self.window.auto_bins.total, 1)
        self.assertIsNone(self.window._job)

    def test_debug_ui_has_no_empty_capture_action(self):
        buttons = [button.text() for button in self.window.findChildren(QPushButton)]
        self.assertFalse(any('采集空' in text for text in buttons))


if __name__ == '__main__':
    unittest.main()
