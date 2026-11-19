import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app import QApplication, QMessageBox
from esp32_serial import SerialLineTransport
from live_console import LiveWindow, COLORS

APP = QApplication.instance() or QApplication([])


class Port:
    def __init__(self):
        self.data = b''
        self.writes = []

    def write(self, data):
        self.writes.extend(data.decode('ascii').splitlines())
        return len(data)

    def read(self, count):
        data, self.data = self.data[:count], self.data[count:]
        return data

    def close(self):
        pass


class FullResetUiTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.window = LiveWindow(data_dir=self.folder.name, butu_24=True)
        self.window.timer.stop()
        self.port = Port()
        self.window.serial_port = self.port
        self.window.transport = SerialLineTransport(self.port)
        self.window.machine.begin_handshake()
        self.reply('PONG', 'VERSION,ESP32_SORTER_XY_PUSHER,0.1.0', 'STATUS,MCU,WAIT_START,0,0,77')
        self.assertTrue(self.window.machine.snapshot().origin_confirmed)
        self.window.machine.goto(40, 30)
        self.reply('STATUS,MCU,WAIT_START,1600,1170,77')
        self.window.save_calibration(self.window.calibration_rows)
        self.window.calibration_confirmed.setChecked(True)
        self.window.vision = SimpleNamespace()
        self.window._frames_at = time.monotonic()
        self.window._models_warm = True
        self.bin_id = self.window.auto_bins.reserve(self.window.names[0], COLORS[0])
        self.window.auto_bins.complete(self.bin_id)
        self.port.writes.clear()

    def tearDown(self):
        self.window.vision = None
        self.window._finish_serial_close()
        self.window.close()
        self.folder.cleanup()

    def reply(self, *lines):
        self.port.data += ('\n'.join(lines) + '\n').encode('ascii')
        self.window.poll_serial()

    def finish_reset(self):
        self.reply('RX,STOP', 'STATUS,PUSHER_UP', 'STATUS,STOP',
                   'RX,STATUS?', 'STATUS,MCU,STOP,1600,1170,77',
                   'RX,PUSHER,UP', 'STATUS,PUSHER_UP',
                   'RX,GOTO_MM,0.000,0.000', 'RX,STATUS?', 'STATUS,MCU,STOP,0,0,77',
                   'RX,RESET', 'READY', 'STATUS,WAIT_START',
                   'RX,STATUS?', 'STATUS,MCU,WAIT_START,0,0,77')

    def test_full_reset_clears_round_only_after_verified_home_and_reset(self):
        before = self.window.auto_bins_path.read_bytes()
        with patch.object(self.window, '_ask_origin_unchanged', return_value=True) as question:
            self.assertTrue(self.window.reset())
            question.assert_called_once_with(full_reset=True)
        self.assertEqual(self.window.auto_bins_path.read_bytes(), before)
        self.assertNotIn('RESET', self.port.writes)
        self.assertTrue(self.window.machine.reset_in_progress)
        self.finish_reset()
        self.assertEqual(self.window.auto_bins.total, 0)
        self.assertEqual(self.window.bindings, {})
        self.assertIsNone(self.window.auto_bins.pending)
        self.assertEqual(self.window.machine.snapshot().state, 'armed')
        self.assertNotIn('HOST_READY', self.port.writes)
        self.assertNotIn('START', self.port.writes)

    def test_cancel_and_reset_failure_preserve_round_and_do_not_start(self):
        before = self.window.auto_bins_path.read_bytes()
        with patch.object(self.window, '_ask_origin_unchanged', return_value=False):
            self.assertFalse(self.window.reset())
        self.assertEqual(self.window.auto_bins_path.read_bytes(), before)
        self.assertNotIn('STOP', self.port.writes)
        with patch.object(self.window, '_ask_origin_unchanged', return_value=True):
            self.assertTrue(self.window.reset())
        self.reply('BOOT,88')
        self.assertEqual(self.window.auto_bins_path.read_bytes(), before)
        self.assertNotIn('START', self.port.writes)

    def test_normal_start_uses_confirmed_origin_without_repeating_question(self):
        cancel = getattr(QMessageBox, 'Cancel', None)
        if cancel is None:
            cancel = QMessageBox.StandardButton.Cancel
        with patch('live_console.QMessageBox.question', return_value=cancel):
            self.assertTrue(self.window.start())
        self.assertIn('HOST_READY', self.port.writes)
        self.assertNotIn('RESET', self.port.writes)

    def test_start_after_normal_stop_rearms_known_origin_without_clearing_round(self):
        before = self.window.auto_bins_path.read_bytes()
        self.window.stop()
        self.reply('STATUS,STOP', 'STATUS,MCU,STOP,1600,1170,77')
        with patch.object(self.window, '_ask_origin_unchanged', return_value=True):
            self.assertTrue(self.window.start())
        self.assertTrue(self.window.machine.reset_in_progress)
        self.assertNotIn('HOST_READY', self.port.writes)
        self.finish_reset()
        self.assertEqual(self.window.auto_bins_path.read_bytes(), before)
        self.assertIn('HOST_READY', self.port.writes)

    def test_untrusted_origin_does_not_rebase_tray_position(self):
        self.window.machine.disconnect()
        self.window.machine.begin_handshake()
        self.reply('PONG', 'VERSION,ESP32_SORTER_XY_PUSHER,0.1.0',
                   'STATUS,MCU,WAIT_START,1600,1170,88')
        self.port.writes.clear()
        with patch.object(self.window, '_ask_origin_unchanged', return_value=True):
            self.assertFalse(self.window.start())
            self.assertFalse(self.window.reset())
        self.assertNotIn('RESET', self.port.writes)
        self.assertFalse(any(command.startswith('GOTO') for command in self.port.writes))

    def test_start_question_is_default_cancel_and_does_not_clear_round(self):
        cancel = getattr(QMessageBox, 'Cancel', None)
        if cancel is None:
            cancel = QMessageBox.StandardButton.Cancel
        before = self.window.auto_bins_path.read_bytes()
        with patch('live_console.QMessageBox.question', return_value=cancel) as question:
            self.assertFalse(self.window._ask_origin_unchanged())
        args, kwargs = question.call_args
        self.assertEqual(kwargs.get('defaultButton', args[4] if len(args) > 4 else None), cancel)
        self.assertEqual(self.window.auto_bins_path.read_bytes(), before)

    def test_declared_origin_change_invalidates_reuse_but_cancel_does_not(self):
        no = getattr(QMessageBox, 'No', None)
        if no is None:
            no = QMessageBox.StandardButton.No
        before = self.window.auto_bins_path.read_bytes()
        with patch('live_console.QMessageBox.question', return_value=no):
            self.assertFalse(self.window._ask_origin_unchanged())
        self.assertFalse(self.window.machine.snapshot().origin_confirmed)
        self.assertFalse(self.window.machine.can_reuse_origin)
        self.assertEqual(self.window.auto_bins_path.read_bytes(), before)
        self.assertNotIn('RESET', self.port.writes)

    def test_full_reset_rejects_new_manual_vision_requests(self):
        self.window.vision.recognize = lambda *a, **kw: True
        self.window.vision.locate = lambda *a, **kw: True
        with patch.object(self.window, '_ask_origin_unchanged', return_value=True):
            self.assertTrue(self.window.reset())
        self.assertFalse(self.window.manual_recognize())
        self.assertFalse(self.window.manual_locate())
        self.assertIsNone(self.window._job)
