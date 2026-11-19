"""Unknown scans discard through the floor without reserving a category bin."""
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from app import QApplication
from live_console import LiveWindow

APP = QApplication.instance() or QApplication([])


class UnknownRejectWindowTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        with patch('side_model_store.SideModelStore.read_active', return_value=None):
            self.window = LiveWindow(data_dir=self.folder.name, linux_20261107=True)
        self.addCleanup(self.window.close)
        self.addCleanup(setattr, self.window, 'vision', None)
        self.addCleanup(self.window.machine.disconnect)
        self.confirm = patch.object(LiveWindow, '_ask_origin_unchanged', return_value=True)
        self.confirm.start()
        self.addCleanup(self.confirm.stop)
        self.now = 100.
        machine = self.window.machine
        machine._clock = lambda: self.now
        self.sent = []
        machine._send = self.sent.append
        machine.begin_handshake()
        for reply in ('PONG', 'VERSION,ESP32_SORTER_XY_PUSHER,0.1.0',
                      'STATUS,MCU,WAIT_START,0,0,77'):
            machine.handle_mcu(reply)
        self.assertTrue(machine.snapshot().origin_confirmed)
        self.window.save_calibration(self.window.calibration_rows)
        self.window.calibration_confirmed.setChecked(True)
        self.window.vision = SimpleNamespace(locate=lambda *a, **kw: True,
                                             recognize=lambda *a, **kw: True)
        self.window._frames_at = time.monotonic()
        self.window._models_warm = True

    def begin_scan(self):
        machine = self.window.machine
        self.assertTrue(self.window.start())
        machine.handle_mcu('STATUS,HOST_READY')
        machine.handle_mcu('STATUS,START')
        machine.handle_mcu('PUSH?')
        self.window._dispatch_vision()
        self.window.on_located(self.window._job[0], (100, 75))
        machine.handle_mcu('SCAN')
        self.sent[:] = []

    def result(self, error=None):
        return SimpleNamespace(error=error, detection=None, color=None, elapsed_ms=10.,
            preview_bgr=np.zeros((320, 640, 3), np.uint8), diagnostics={},
            mask_status=('disabled', 'disabled'))

    def observe_unknown(self):
        self.window._scan_next_at = 0
        self.window._dispatch_vision()
        self.window.on_recognized(self.window._job[0], self.result())

    def test_release_exhausted_unknown_opens_floor_without_stop_sort_or_bin(self):
        self.begin_scan()
        for _ in range(3):
            self.observe_unknown()
        self.assertEqual(self.window.machine.snapshot().state, 'reject_opening')
        self.assertEqual(self.sent, ['SERVO2,OPEN'])
        self.assertEqual(self.window.bindings, {})
        self.assertEqual(self.window.auto_bins.total, 0)
        self.assertIsNone(self.window.auto_bins.pending)
        self.assertIn('未知', self.window._release_status_values(self.window.machine.snapshot())[0])

    def test_next_piece_gets_fresh_consensus_after_confirmed_unknown_discard(self):
        self.begin_scan()
        for _ in range(3):
            self.observe_unknown()
        machine = self.window.machine
        self.assertEqual(machine.snapshot().state, 'reject_opening')
        machine.handle_mcu('RX,SERVO2,OPEN')
        machine.handle_mcu('STATUS,SORT_GATE_OPEN')
        self.now += 1.
        machine.tick()
        machine.handle_mcu('RX,SERVO2,CLOSE')
        machine.handle_mcu('STATUS,SORT_GATE_CLOSE')
        self.now += 1.
        machine.tick()
        self.assertEqual(self.sent, ['SERVO2,OPEN', 'SERVO2,CLOSE', 'PUSH,WAIT'])
        machine.handle_mcu('STATUS,PUSH,WAIT')
        machine.handle_mcu('PUSH?')
        self.window._dispatch_vision()
        self.window.on_located(self.window._job[0], (120, 60))
        machine.handle_mcu('SCAN')
        self.window._dispatch_vision()
        self.assertEqual(self.window._scan_gate.attempts, 0)
        self.assertEqual(self.window.machine.snapshot().completed_count, 0)
        self.assertEqual(self.window.auto_bins.total, 0)
        self.assertFalse(any(command.startswith('SORT,') for command in self.sent))

    def test_manual_unknown_never_opens_floor(self):
        self.sent[:] = []
        self.assertTrue(self.window.manual_recognize())
        self.window.on_recognized(self.window._job[0], self.result())
        self.assertEqual(self.sent, [])
        self.assertEqual(self.window.machine.snapshot().state, 'armed')

    def test_camera_failure_stops_instead_of_opening_floor(self):
        self.begin_scan()
        self.window._dispatch_vision()
        self.window.on_recognized(self.window._job[0], self.result('camera disconnected'))
        self.assertIn('STOP', self.sent)
        self.assertNotIn('SERVO2,OPEN', self.sent)

    def test_expired_scan_never_opens_floor(self):
        self.begin_scan()
        self.observe_unknown()
        self.observe_unknown()
        self.window._scan_next_at = 0
        self.window._dispatch_vision()
        token = self.window._job[0]
        self.now += 5.
        self.window.on_recognized(token, self.result())
        self.assertIn('STOP', self.sent)
        self.assertNotIn('SERVO2,OPEN', self.sent)


if __name__ == '__main__':
    unittest.main()
