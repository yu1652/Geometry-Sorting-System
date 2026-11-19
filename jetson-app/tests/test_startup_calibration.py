import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from app import QApplication, QMessageBox
from live_console import LiveWindow
from live_control import LiveController


APP = QApplication.instance() or QApplication([])
POINTS = [(f'P{row * 3 + column + 1}', 211 + column * 133, 92 + row * 133,
           24 + column * 56, 24 + row * 56)
          for row in range(3) for column in range(3)]


class Vision:
    def shutdown(self):
        pass


class StartupCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.data_dir = Path(self.folder.name)
        self.windows = []
        self.addCleanup(self.close_windows)

    def close_windows(self):
        for window in reversed(self.windows):
            window.vision = None
            window._finish_serial_close()
            window.close()

    def window(self, **settings):
        window = LiveWindow(data_dir=self.data_dir, **settings)
        window.timer.stop()
        self.windows.append(window)
        return window

    def saved_points(self, points=POINTS):
        path = self.data_dir / 'live_calibration.json'
        path.write_text(json.dumps({'mode': 'live', 'points': points}), encoding='utf8')
        return path

    def ready_window(self, confirm_origin=True):
        window = self.window()
        window.save_calibration(POINTS)
        window.calibration_confirmed.setChecked(True)
        window.linux_20261107 = True
        window.vision = Vision()
        window._frames_at = time.monotonic()
        window._models_warm = True
        commands = []
        window.machine = LiveController(commands.append)
        window.machine.begin_handshake()
        for line in ('PONG', 'VERSION,ESP32_SORTER_XY_PUSHER,0.1.0',
                     'STATUS,MCU,WAIT_START,0,0,77'):
            window.machine.handle_mcu(line)
        if confirm_origin:
            self.assertTrue(window.machine.confirm_origin())
            window.machine.handle_mcu('READY')
        commands.clear()
        return window, commands

    def test_saved_nine_points_load_ready_without_manual_check(self):
        path = self.saved_points()
        before = path.read_bytes()
        window = self.window()
        self.assertTrue(window.calibration_valid)
        self.assertTrue(window.calibration_confirmed.isChecked())
        self.assertFalse(window.calibration_confirmed.isEnabled())
        self.assertEqual(window.calibration_rows, [list(row) for row in POINTS])
        self.assertEqual(path.read_bytes(), before)
        self.assertIsNone(window.vision)
        self.assertIsNone(window.serial_port)
        self.assertFalse(window.machine.snapshot().origin_confirmed)

    def test_startup_opens_fixed_devices_as_they_appear_without_starting_sorting(self):
        serial_path = self.data_dir / 'fixed-esp32'
        sources = {role: str(self.data_dir / role) for role in ('left', 'right', 'top')}
        (self.data_dir / 'live_devices.json').write_text(json.dumps({'sources': sources}), encoding='utf8')
        window = self.window(startup_serial_port=str(serial_path))
        with patch.object(window, 'connect_serial') as serial_open, \
                patch.object(window, 'connect_cameras') as cameras_open, \
                patch.object(window, 'start') as start:
            window._connect_startup_devices()
            serial_open.assert_not_called()
            cameras_open.assert_not_called()

            serial_path.touch()
            window._connect_startup_devices()
            serial_open.assert_called_once_with()
            cameras_open.assert_not_called()
            self.assertEqual(window.port_picker.currentText(), str(serial_path))

            for path in sources.values():
                Path(path).touch()
            window._connect_startup_devices()
            window._connect_startup_devices()
            serial_open.assert_called_once_with()
            cameras_open.assert_called_once_with()
            start.assert_not_called()

    def test_startup_timeout_does_not_fall_back_to_another_device(self):
        path = self.data_dir / 'missing-esp32'
        window = self.window(startup_serial_port=str(path))
        window.sources = {role: str(self.data_dir / role) for role in ('left', 'right', 'top')}
        window._startup_deadline = 0
        with patch.object(window, 'connect_serial') as serial_open, \
                patch.object(window, 'connect_cameras') as cameras_open, \
                patch.object(window, 'write_log') as log:
            window._connect_startup_devices()
            log.assert_called_once()
            self.assertIn(str(path), log.call_args[0][0])
            self.assertIn(window.sources['top'], log.call_args[0][0])
            path.touch()
            window._connect_startup_devices()
            serial_open.assert_not_called()
            cameras_open.assert_not_called()

    def test_new_boot_starts_empty_and_reopening_preserves_current_round(self):
        previous = self.window()
        previous.save_calibration(POINTS)
        shape = previous.names[0]
        previous.auto_bins.complete(previous.auto_bins.reserve(shape, '红色'))
        previous.auto_bins.reserve(shape, '蓝色')

        boot = self.window(round_boot_id='boot-one')
        self.assertTrue(boot.calibration_valid)
        self.assertEqual(boot.auto_bins.counts, (0,) * 6)
        self.assertEqual(boot.auto_bins.bindings, {})
        self.assertIsNone(boot.auto_bins.pending)
        self.assertEqual(boot.auto_bins.reserve(shape, '蓝色'), 1)
        boot.auto_bins.complete(1)

        reopened = self.window(round_boot_id='boot-one')
        self.assertEqual(reopened.auto_bins.counts, (1, 0, 0, 0, 0, 0))
        self.assertEqual(reopened.auto_bins.bindings, {(shape, '蓝色'): 1})

        restarted = self.window(round_boot_id='boot-two')
        self.assertEqual(restarted.auto_bins.counts, (0,) * 6)
        self.assertEqual(restarted.auto_bins.bindings, {})

    def test_saving_valid_nine_points_is_ready_immediately(self):
        window = self.window()
        window.save_calibration(POINTS)
        self.assertTrue(window.calibration_valid)
        self.assertTrue(window.calibration_confirmed.isChecked())
        restored = self.window()
        self.assertTrue(restored.calibration_confirmed.isChecked())

    def test_competition_profile_calibration_is_automatically_checked_and_read_only(self):
        self.saved_points()
        window = self.window(butu_24=True)
        self.assertTrue(window.calibration_valid)
        self.assertTrue(window.calibration_confirmed.isChecked())
        self.assertFalse(window.calibration_confirmed.isEnabled())

    def test_missing_file_does_not_authorize_default_example_points(self):
        window = self.window()
        self.assertFalse(window.calibration_valid)
        self.assertFalse(window.calibration_confirmed.isChecked())
        self.assertFalse(window.start())

    def test_invalid_json_does_not_authorize_motion(self):
        (self.data_dir / 'live_calibration.json').write_text('{', encoding='utf8')
        window = self.window()
        self.assertFalse(window.calibration_valid)
        self.assertFalse(window.calibration_confirmed.isChecked())
        self.assertFalse(window.start())

    def test_out_of_range_saved_points_are_rejected_not_clamped(self):
        points = [list(row) for row in POINTS]
        points[0][3] = 193.501
        path = self.saved_points(points)
        before = path.read_bytes()
        window = self.window()
        self.assertFalse(window.calibration_valid)
        self.assertFalse(window.calibration_confirmed.isChecked())
        self.assertEqual(path.read_bytes(), before)

    def test_invalid_save_preserves_previously_valid_calibration(self):
        window = self.window()
        window.save_calibration(POINTS)
        before = window.calibration_path.read_bytes()
        points = [list(row) for row in POINTS]
        points[0][4] = 143.001
        with self.assertRaises(ValueError):
            window.save_calibration(points)
        self.assertEqual(window.calibration_path.read_bytes(), before)
        self.assertTrue(window.calibration_confirmed.isChecked())

    def test_confirmed_origin_starts_without_repeating_question_or_reset(self):
        window, commands = self.ready_window()
        cancel = getattr(QMessageBox, 'Cancel', None)
        if cancel is None:
            cancel = QMessageBox.StandardButton.Cancel
        with patch('live_console.QMessageBox.question', return_value=cancel):
            self.assertTrue(window.start())
        self.assertEqual(commands, ['HOST_READY'])
        self.assertEqual(window.machine.snapshot().state, 'enabling_auto')

    def test_physical_press_uses_the_same_start_handshake_once(self):
        window, commands = self.ready_window()
        lines = ['STATUS,WAIT_HOST', 'STATUS,WAIT_HOST']
        window.transport = SimpleNamespace(read_lines=lambda: lines)
        window.poll_serial()
        self.assertEqual(commands, ['HOST_READY'])
        lines[:] = ['STATUS,HOST_READY', 'STATUS,START']
        window.poll_serial()
        self.assertEqual(commands, ['HOST_READY', 'START'])
        self.assertEqual(window.machine.snapshot().state, 'running')

    def test_start_status_waits_for_models_and_frames_then_acknowledges_one_press(self):
        window, commands = self.ready_window()

        def status():
            return window._release_status_values(window.machine.snapshot())[0]

        window._models_warm = False
        window._frames_at = None
        self.assertIn('模型预热中', status())
        window._models_warm = True
        self.assertIn('等待三路相机画面', status())
        window._frames_at = time.monotonic()
        self.assertEqual(status(), '设备就绪，可按实体启动键')
        window.transport = SimpleNamespace(read_lines=lambda: ['STATUS,WAIT_HOST'])
        window.poll_serial()
        self.assertEqual(commands, ['HOST_READY'])
        self.assertEqual(status(), '启动中，等待控制板确认')

    def test_rejected_serial_start_is_not_a_physical_press(self):
        window, commands = self.ready_window()
        lines = ['RX,START']
        window.transport = SimpleNamespace(read_lines=lambda: lines)
        window.poll_serial()
        lines[:] = ['STATUS,WAIT_HOST']
        window.poll_serial()
        self.assertEqual(commands, [])

    def test_early_physical_press_expires_without_starting_later(self):
        window, commands = self.ready_window()
        window._models_warm = False
        lines = ['STATUS,WAIT_HOST']
        window.transport = SimpleNamespace(read_lines=lambda: lines)
        window.poll_serial()
        self.assertEqual(commands, [])
        window._models_warm = True
        lines.clear()
        window.poll_serial()
        self.assertEqual(commands, [])

    def test_physical_press_cannot_confirm_an_unconfirmed_origin(self):
        window, commands = self.ready_window(confirm_origin=False)
        window.transport = SimpleNamespace(read_lines=lambda: ['STATUS,WAIT_HOST'])
        with patch('live_console.QMessageBox.question') as question:
            window.poll_serial()
        question.assert_not_called()
        self.assertEqual(commands, [])

    def test_cold_start_does_not_treat_reported_zero_as_confirmed_origin(self):
        window, commands = self.ready_window(confirm_origin=False)
        self.assertFalse(window.start())
        self.assertEqual(commands, [])
        self.assertFalse(window.machine.snapshot().origin_confirmed)

    def test_controller_reboot_still_requires_actual_origin_confirmation(self):
        window, commands = self.ready_window()
        window.machine.handle_mcu('BOOT,88')
        for line in ('PONG', 'VERSION,ESP32_SORTER_XY_PUSHER,0.1.0',
                     'STATUS,MCU,WAIT_START,0,0,88'):
            window.machine.handle_mcu(line)
        commands.clear()
        self.assertFalse(window.start())
        self.assertEqual(commands, [])
        self.assertFalse(window.machine.snapshot().origin_confirmed)

    def test_explicit_origin_invalidation_is_not_bypassed(self):
        window, commands = self.ready_window()
        window.machine.invalidate_origin()
        window.machine.handle_mcu('STATUS,MCU,STOP,0,0,77')
        commands.clear()
        self.assertFalse(window.start())
        self.assertEqual(commands, [])


if __name__ == '__main__':
    unittest.main()
