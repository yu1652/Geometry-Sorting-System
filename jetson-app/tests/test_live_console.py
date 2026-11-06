import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import app
import numpy as np

APP = app.QApplication.instance() or app.QApplication([])


class FakePort:
    def __init__(self):
        self.data = b''
        self.writes = []
        self.closed = False

    def read(self, size):
        data, self.data = self.data[:size], self.data[size:]
        return data

    def write(self, data):
        self.writes.append(data)
        return len(data)

    def close(self):
        self.closed = True


class FakeVision:
    def recognize(self, token):
        return True

    def locate(self, *args):
        return True

    def shutdown(self):
        pass


class LiveConsoleTests(unittest.TestCase):
    def setUp(self):
        from live_console import LiveWindow
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.window = LiveWindow(data_dir=Path(self.folder.name))
        self.addCleanup(self.window.close)

    def test_default_has_two_pages_and_no_connection_or_sample_calibration_authority(self):
        window = self.window
        self.assertEqual(window.pages.count(), 2)
        self.assertIn('实机', window.windowTitle())
        self.assertIsNone(window.vision)
        self.assertIsNone(window.serial_port)
        self.assertFalse(window.calibration_confirmed.isChecked())
        self.assertFalse(window.machine.snapshot().origin_confirmed)
        self.assertFalse(window.start())
        for name in ('运行', '复位', '退出'):
            self.assertEqual(window.actions[name].text(), name)
        window.show_competition()
        self.assertIs(window.pages.currentWidget(), window.competition_page)
        window.show_debug()
        self.assertIs(window.pages.currentWidget(), window.debug_page)

    def test_actual_camera_role_order_and_saved_calibration_still_requires_validation(self):
        window = self.window
        self.assertIn('2.1:', window.sources['left'])
        self.assertIn('2.2:', window.sources['right'])
        self.assertIn('2.4:', window.sources['top'])
        window.save_calibration(window.calibration_rows)
        value = json.loads(window.calibration_path.read_text(encoding='utf8'))
        self.assertEqual(value['mode'], 'live')
        self.assertEqual(len(value['points']), 9)
        self.assertFalse(window.calibration_confirmed.isChecked())
        dialog = window.create_calibration_dialog()
        self.addCleanup(dialog.reject)
        self.assertNotIn('仿真', dialog.windowTitle())
        self.assertEqual(dialog.points.rowCount(), 9)

    def test_machine_range_is_independent_of_tray_size_and_survives_save_reload(self):
        from sorter_console import CoordinateSpinBox, calibration_rows
        from live_console import LiveWindow, validate_points
        dialog = self.window.create_calibration_dialog()
        self.addCleanup(dialog.reject)
        x = dialog.findChild(CoordinateSpinBox, 'coordinate_2_3')
        y = dialog.findChild(CoordinateSpinBox, 'coordinate_8_4')
        self.assertEqual(x.maximum(), 260)
        self.assertEqual(y.maximum(), 230)
        for row in range(9):
            dialog.points.item(row, 3).setText(str(70 + 56 * (row % 3)))
            dialog.points.item(row, 4).setText(str(12 + 56 * (row // 3)))
        rows = calibration_rows(dialog.points)
        mapping = validate_points(rows)
        self.assertAlmostEqual(mapping.map_pixel(477, 92)[0], 182)
        mapping.validate_physical(260, 230)
        for position in ((260.001, 230), (260, 230.001), (-.001, 0)):
            with self.assertRaises(ValueError):
                mapping.validate_physical(*position)
        dialog.set_frame((150.43, 47.86, 377.44, 374.7))
        dialog.frame_inset.setValue(18)
        dialog.save_points()
        restored = LiveWindow(data_dir=self.window.data_dir)
        self.addCleanup(restored.close)
        self.assertTrue(restored.calibration_valid)
        self.assertEqual(restored.calibration_rows[2][3], 182)
        saved = restored.create_calibration_dialog()
        self.addCleanup(saved.reject)
        self.assertEqual(saved.view.box, dialog.view.box)
        self.assertEqual(saved.frame_inset.value(), 18)
        self.assertFalse(restored.calibration_confirmed.isChecked())

    def test_out_of_travel_calibration_does_not_overwrite_saved_points(self):
        from sorter_console import DEFAULT_POINTS
        self.window.save_calibration(list(DEFAULT_POINTS))
        before = self.window.calibration_path.read_bytes()
        for x, y in ((260.001, 100), (100, 230.001), (float('nan'), 20)):
            rows = [list(row) for row in DEFAULT_POINTS]
            rows[0][3:5] = (x, y)
            with self.subTest(x=x, y=y), self.assertRaises(ValueError):
                self.window.save_calibration(rows)
            self.assertEqual(self.window.calibration_path.read_bytes(), before)

    def test_selected_calibration_goto_above_160_keeps_real_ack_requirement(self):
        from sorter_console import CoordinateSpinBox
        sent = self.ready_manual_controller()
        dialog = self.window.create_calibration_dialog()
        self.addCleanup(dialog.reject)
        before = list(sent)
        dialog.findChild(CoordinateSpinBox, 'coordinate_2_3').setValue(182)
        dialog.findChild(CoordinateSpinBox, 'coordinate_2_4').setValue(12)
        self.assertEqual(sent, before)
        self.assertEqual(self.window.machine.snapshot().x_mm, 0)
        dialog.points.setCurrentCell(2, 3)
        dialog.goto_selected()
        self.assertEqual(sent[-2:], ['GOTO_MM,182.000,12.000', 'STATUS?'])
        self.assertEqual(self.window.machine.pending, 'goto')
        self.window.machine.handle_mcu('STATUS,MCU,WAIT_START,7280,468,77')
        self.assertFalse(self.window.machine.pending)
        self.assertEqual(self.window.machine.snapshot().x_mm, 182)

    def test_invalid_frame_cannot_overwrite_saved_calibration(self):
        self.window.save_calibration(self.window.calibration_rows)
        before = self.window.calibration_path.read_bytes()
        for frame in ({'box': [600, 20, 200, 100], 'inset_percent': 15},
                      {'box': [100, 20, 200, 100], 'inset_percent': float('nan')}):
            with self.subTest(frame=frame), self.assertRaises(ValueError):
                self.window.save_calibration(self.window.calibration_rows, frame=frame)
            self.assertEqual(self.window.calibration_path.read_bytes(), before)

    def test_existing_six_adjusted_points_survive_old_file_upgrade_without_confirmation(self):
        from sorter_console import DEFAULT_POINTS, calibration_rows
        from live_console import LiveWindow
        rows = [list(row) for row in DEFAULT_POINTS]
        confirmed = {0:(70,12), 1:(126,12), 3:(70,71), 4:(126,71),
                     6:(71,127), 7:(128.5,127.5)}
        for index, position in confirmed.items():
            rows[index][3:5] = position
        rows[2][3:5] = (159,12)  # Other points still awaiting physical adjustment.
        self.window.calibration_path.write_text(json.dumps({'mode':'live','points':rows}), encoding='utf8')
        restored = LiveWindow(data_dir=self.window.data_dir)
        self.addCleanup(restored.close)
        dialog = restored.create_calibration_dialog()
        self.addCleanup(dialog.reject)
        self.assertEqual(calibration_rows(dialog.points), [tuple(row) for row in rows])
        self.assertFalse(restored.calibration_confirmed.isChecked())
        dialog.set_frame((150.43,47.86,377.44,374.7))
        dialog.save_points()
        self.assertEqual(restored.calibration_rows, [tuple(row) for row in rows])
        self.assertFalse(restored.calibration_confirmed.isChecked())

    def test_manual_result_does_not_count_and_stale_result_is_discarded(self):
        window = self.window
        result = SimpleNamespace(
            error=None, detection=SimpleNamespace(class_name='yuan_zhu', confidence=.9),
            color=SimpleNamespace(name='蓝色', confidence=.8, valid_pixels=120),
            preview_bgr=np.zeros((320, 640, 3), np.uint8), elapsed_ms=100,
        )
        window._job = (1, window.machine.epoch, 'manual')
        window.on_recognized(1, result)
        self.assertIn('yuan_zhu', window.result_label.text())
        self.assertEqual(window.records.rowCount(), 0)
        self.assertEqual(window.machine.snapshot().completed_count, 0)
        window._job = (2, window.machine.epoch, 'manual')
        window.machine.disconnect()
        window.result_label.setText('新状态')
        window.on_recognized(2, result)
        self.assertEqual(window.result_label.text(), '新状态')

    def test_surface_participation_and_hole_candidates_are_visible(self):
        from legacy_vision import LegacyResult
        result = LegacyResult(
            detection=SimpleNamespace(class_name='wu_leng_huan_zhu', confidence=.8),
            color=SimpleNamespace(name='蓝色', confidence=.9, valid_pixels=1000),
            mask_status=('applied', 'empty'), preview_bgr=np.zeros((320, 640, 3), np.uint8),
            elapsed_ms=200, hole_counts=(1, None), color_sources=('mask', 'box'))
        self.window._job = (1, self.window.machine.epoch, 'manual')
        self.window.on_recognized(1, result)
        self.assertIn('左覆膜：已参与', self.window.result_label.text())
        self.assertIn('孔候选1', self.window.result_label.text())
        self.assertIn('右覆膜：空掩膜', self.window.result_label.text())
        self.assertEqual(self.window.records.rowCount(), 0)

    def test_binding_validation_rejects_duplicate_and_unknown_class_before_write(self):
        window = self.window
        rows = [{'shape': 'yuan_zhu', 'color': '蓝色', 'bin': 1}]
        window.save_bindings(rows)
        before = window.bindings_path.read_bytes()
        with self.assertRaises(ValueError):
            window.save_bindings(rows + [dict(rows[0], bin=2)])
        self.assertEqual(window.bindings_path.read_bytes(), before)
        with self.assertRaises(ValueError):
            window.save_bindings([dict(rows[0], shape='wrong_class')])

    def ready_manual_controller(self):
        from live_control import LiveController
        sent = []
        controller = self.window.machine = LiveController(sent.append)
        controller.begin_handshake()
        for line in ('PONG', 'VERSION,ESP32_SORTER_XY_PUSHER,0.1.0', 'STATUS,MCU,WAIT_START,0,0,77'):
            controller.handle_mcu(line)
        self.assertTrue(controller.confirm_origin())
        controller.handle_mcu('READY')
        self.assertEqual(controller.snapshot().state, 'armed')
        return sent

    def test_frame_draft_keeps_live_calibration_and_commands_unchanged_until_save(self):
        from sorter_console import calibration_rows
        sent = self.ready_manual_controller()
        window = self.window
        window.save_calibration(window.calibration_rows)
        window.calibration_confirmed.setChecked(True)
        saved = window.calibration_path.read_bytes()
        sent.clear()
        dialog = window.create_calibration_dialog()
        self.addCleanup(dialog.reject)
        dialog.frame_fields[0].setValue(180)
        dialog.frame_fields[2].setValue(300)
        dialog.frame_inset.setValue(20)
        dialog.generate_points()
        self.assertEqual(calibration_rows(dialog.points)[0][1], 240)
        self.assertEqual(sent, [])
        self.assertEqual(window.calibration_path.read_bytes(), saved)
        self.assertTrue(window.calibration_confirmed.isChecked())
        dialog.save_points()
        self.assertFalse(window.calibration_confirmed.isChecked())
        self.assertEqual(window.calibration_rows[0][1], 240)
        self.assertEqual(sent, [])

    def test_calibration_preview_refreshes_new_camera_frames_and_stops_on_close(self):
        try:
            from PySide6.QtTest import QTest
        except ImportError:
            from PyQt5.QtTest import QTest
        dialog = self.window.create_calibration_dialog()
        self.addCleanup(dialog.reject)
        dialog.show()
        self.window.frames['top'] = np.full((480, 640, 3), 37, dtype=np.uint8)
        QTest.qWait(220)
        self.assertEqual(dialog.view.frame[470, 630].tolist(), [37, 37, 37])
        dialog.reject()
        self.assertFalse(dialog.preview_timer.isActive())

    def test_closing_calibration_while_down_is_pending_queues_up(self):
        sent = self.ready_manual_controller()
        dialog = self.window.create_calibration_dialog()
        dialog.move_pusher(True)
        self.assertEqual(self.window.machine.pending, 'down')
        self.assertFalse(self.window.machine.snapshot().pusher_down)
        dialog.reject()
        self.assertEqual(sent[-1], 'PUSHER,UP')
        self.window.machine.handle_mcu('STATUS,PUSHER_DOWN')
        self.assertEqual(self.window.machine.pending, 'up')
        self.window.machine.handle_mcu('STATUS,PUSHER_UP')
        self.assertIsNone(self.window.machine.pending)

    def test_disconnect_keeps_port_open_until_real_stop_ack(self):
        from esp32_serial import SerialLineTransport
        self.ready_manual_controller()
        port = self.window.serial_port = FakePort()
        self.window.transport = SerialLineTransport(port)
        self.addCleanup(self.window._finish_serial_close)
        self.window.close_serial()
        self.assertFalse(port.closed)
        port.data = b'STATUS,MCU,STOP,bad\n'
        self.window.poll_serial()
        self.assertFalse(port.closed)
        port.data = b'STATUS,STOP\n'
        self.window.poll_serial()
        self.assertTrue(port.closed)
        self.assertIsNone(self.window.serial_port)
        self.assertFalse(self.window.machine.snapshot().origin_confirmed)

    def test_reset_waits_for_stop_ack_before_prompting_for_manual_origin(self):
        from esp32_serial import SerialLineTransport
        window = self.window
        sent = self.ready_manual_controller()
        sent.clear()
        port = window.serial_port = FakePort()
        window.transport = SerialLineTransport(port)
        self.addCleanup(window._finish_serial_close)
        with patch.object(window, 'confirm_origin') as confirm:
            self.assertTrue(window.reset())
            self.assertNotIn('RESET', sent)
            self.assertFalse(confirm.called)
            port.data = b'STATUS,STOP\n'
            window.poll_serial()
            APP.processEvents()
            confirm.assert_called_once()

    def test_cannot_reconnect_serial_while_closing(self):
        with patch.object(self.window, '_serial_factory') as factory:
            with patch.object(self.window, '_closing', True):
                self.window.connect_serial()
            factory.assert_not_called()

    def test_stop_cancels_already_queued_reset_prompt(self):
        self.ready_manual_controller()
        epoch = self.window.machine.epoch
        self.window.stop()
        with patch.object(self.window, 'confirm_origin') as confirm:
            self.window._confirm_reset(epoch)
            confirm.assert_not_called()

    def test_invalid_saved_calibration_cannot_authorize_default_points(self):
        from live_console import LiveWindow
        self.window.calibration_path.write_text('{"mode":"live","points":[]}', encoding='utf8')
        restored = LiveWindow(data_dir=self.window.data_dir)
        self.addCleanup(restored.close)
        self.assertFalse(restored.calibration_valid)
        restored.calibration_confirmed.setChecked(True)
        restored.vision = FakeVision()
        restored._frames_at = time.monotonic()
        self.addCleanup(setattr, restored, 'vision', None)
        self.assertFalse(restored.manual_locate())
        self.assertIsNone(restored._job)

    def test_vision_submit_failure_clears_job(self):
        def reject_request(token):
            raise RuntimeError('submit failed')
        self.window.vision = SimpleNamespace(recognize=reject_request)
        self.window._frames_at = time.monotonic()
        self.addCleanup(setattr, self.window, 'vision', None)
        self.assertFalse(self.window.manual_recognize())
        self.assertIsNone(self.window._job)

    def test_models_force_raw_mode_and_block_reconnection(self):
        window = self.window
        process = SimpleNamespace(poll=lambda: None)
        with patch('live_console.subprocess.Popen', return_value=process) as spawn:
            window.open_models()
        self.assertIn('--raw', spawn.call_args[0][0])
        window.connect_cameras()
        window.connect_serial()
        self.assertIsNone(window.vision)
        self.assertIsNone(window.serial_port)

    def test_automatic_cycle_counts_only_real_done(self):
        from esp32_serial import SerialLineTransport
        window = self.window
        sent = self.ready_manual_controller()
        window.vision = FakeVision()
        self.addCleanup(setattr, window, 'vision', None)
        window._frames_at = time.monotonic()
        window.save_calibration(window.calibration_rows)
        window.calibration_confirmed.setChecked(True)
        window.save_bindings([{'shape': 'yuan_zhu', 'color': '蓝色', 'bin': 1}])
        self.assertTrue(window.start())
        window.machine.handle_mcu('STATUS,HOST_READY')
        window.machine.handle_mcu('STATUS,START')
        window.machine.handle_mcu('PUSH?')
        window._dispatch_vision()
        window.on_located(window._job[0], (50, 50))
        window.machine.handle_mcu('SCAN')
        window._dispatch_vision()
        result = SimpleNamespace(error=None,
            detection=SimpleNamespace(class_name='yuan_zhu', confidence=.9),
            color=SimpleNamespace(name='蓝色', confidence=.8, valid_pixels=100),
            preview_bgr=np.zeros((320, 640, 3), np.uint8), elapsed_ms=100)
        window.on_recognized(window._job[0], result)
        self.assertIn('SORT,1', sent)
        self.assertEqual(window.records.rowCount(), 0)
        port = window.serial_port = FakePort()
        window.transport = SerialLineTransport(port)
        self.addCleanup(window._finish_serial_close)
        port.data = b'DONE\nDONE\n'
        window.poll_serial()
        self.assertEqual(window.records.rowCount(), 1)
        self.assertEqual(window.total_label.text(), '本次成功总数 1')
        self.assertTrue((window.data_dir / 'live_results.jsonl').is_file())


if __name__ == '__main__':
    unittest.main()
