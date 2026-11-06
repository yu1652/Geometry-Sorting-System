import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app import QApplication, QMessageBox
from live_console import LiveWindow
from live_control import LiveController
import numpy as np


APP = QApplication.instance() or QApplication([])


class Port:
    def __init__(self):
        self.data = b''
    def read(self, size):
        value, self.data = self.data[:size], self.data[size:]
        return value
    def write(self, data):
        return len(data)
    def close(self):
        pass


class Vision:
    def locate(self, *args):
        return True
    def recognize(self, *args, **kwargs):
        return True
    def shutdown(self):
        pass


class AutomaticBinUiTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.window = LiveWindow(data_dir=Path(self.folder.name))
        self.addCleanup(self.window.close)
        self.sent = []
        machine = self.window.machine = LiveController(self.sent.append)
        machine.begin_handshake()
        for line in ('PONG','VERSION,ESP32_SORTER_XY_PUSHER,0.1.0','STATUS,MCU,WAIT_START,0,0,77'):
            machine.handle_mcu(line)
        machine.confirm_origin()
        machine.handle_mcu('READY')
        self.window.vision = Vision()
        self.addCleanup(setattr, self.window, 'vision', None)
        self.window._frames_at = time.monotonic()
        self.window.save_calibration(self.window.calibration_rows)
        self.window.calibration_confirmed.setChecked(True)
        from esp32_serial import SerialLineTransport
        self.port = self.window.serial_port = Port()
        self.window.transport = SerialLineTransport(self.port)
        self.addCleanup(self.window._finish_serial_close)

    def start_round(self):
        self.assertTrue(self.window.start())
        self.window.machine.handle_mcu('STATUS,HOST_READY')
        self.window.machine.handle_mcu('STATUS,START')

    def recognize(self, shape, color, confidence=.9, manual=False):
        result = SimpleNamespace(error=None,
            detection=SimpleNamespace(class_name=shape, confidence=.9),
            color=SimpleNamespace(name=color, confidence=confidence, valid_pixels=100),
            preview_bgr=np.zeros((320,640,3), np.uint8), elapsed_ms=100)
        if manual:
            self.window._job = (999, self.window.machine.epoch, 'manual')
            self.window.on_recognized(999, result)
            return
        self.window.machine.handle_mcu('PUSH?')
        self.window._dispatch_vision()
        self.window.on_located(self.window._job[0], (50,50))
        self.window.machine.handle_mcu('SCAN')
        self.window._dispatch_vision()
        self.window.on_recognized(self.window._job[0], result)
        if self.window.machine.snapshot().state == 'recognizing':
            self.window._scan_next_at = 0
            self.window._dispatch_vision()
            self.window.on_recognized(self.window._job[0], result)

    def arrive_for_scan(self):
        self.start_round()
        self.window.machine.handle_mcu('PUSH?')
        self.window._dispatch_vision()
        self.window.on_located(self.window._job[0], (50, 50))
        self.window.machine.handle_mcu('SCAN')
        self.window._dispatch_vision()

    def result(self, shape='yuan_zhu', color='蓝色'):
        return SimpleNamespace(error=None,
            detection=SimpleNamespace(class_name=shape, confidence=.9) if shape else None,
            color=SimpleNamespace(name=color, confidence=.9, valid_pixels=100) if color else None,
            preview_bgr=np.zeros((320,640,3), np.uint8), elapsed_ms=100)

    def finish_scan(self, result):
        if self.window._job is None:
            self.window._scan_next_at = 0
            self.window._dispatch_vision()
        self.window.on_recognized(self.window._job[0], result)

    def test_empty_early_frame_is_recaptured_before_binding_and_two_matching_reads_sort_once(self):
        self.arrive_for_scan()
        self.finish_scan(self.result(None, None))
        self.assertEqual(self.window.machine.snapshot().state, 'recognizing')
        self.assertEqual(self.window.bindings, {})
        self.finish_scan(self.result())
        self.assertEqual(self.window.bindings, {})
        self.assertFalse(any(line.startswith('SORT,') for line in self.sent))
        self.finish_scan(self.result())
        self.assertEqual([s for s in self.sent if s.startswith('SORT,')], ['SORT,1'])

    def test_changed_shape_resets_stability_and_stop_discards_pending_result(self):
        self.arrive_for_scan()
        self.finish_scan(self.result('yuan_tai'))
        self.finish_scan(self.result('yuan_zhu'))
        self.assertEqual(self.window.bindings, {})
        self.window._scan_next_at = 0
        self.window._dispatch_vision()
        token = self.window._job[0]
        self.window.stop()
        self.window.on_recognized(token, self.result('yuan_zhu'))
        self.assertFalse(any(s.startswith('SORT,') for s in self.sent))
        self.assertEqual(self.window.bindings, {})

    def test_eight_invalid_reads_stop_without_sort_and_are_available_in_diagnostic_export(self):
        from zipfile import ZipFile
        self.arrive_for_scan()
        for _ in range(8):
            self.finish_scan(self.result(None, None))
        self.assertEqual(self.window.machine.snapshot().state, 'stopped')
        self.assertEqual(self.window.bindings, {})
        self.assertFalse(any(s.startswith('SORT,') for s in self.sent))
        with patch.object(QMessageBox, 'information'):
            path = self.window.export_diagnostics()
        with ZipFile(path) as bundle:
            log = bundle.read('live_runtime.log').decode('utf8')
            self.assertIn('8/8', log)
            self.assertTrue(any(n.startswith('diagnostic_samples/') for n in bundle.namelist()))

    def test_automatic_waits_for_settlement_manual_uses_same_recognizer_without_settlement(self):
        with patch.object(self.window.vision, 'recognize', return_value=True) as recognize:
            self.arrive_for_scan()
            recognize.assert_called_once_with(self.window._job[0], settle=True)
        self.window.stop()
        self.window._job = None
        with patch.object(self.window.vision, 'recognize', return_value=True) as recognize:
            self.assertTrue(self.window.manual_recognize())
            recognize.assert_called_once_with(self.window._job[0])

    def test_model_initialization_can_take_more_than_preview_limit_but_action_timeout_remains(self):
        self.arrive_for_scan()
        before = time.monotonic()
        self.window.machine._clock = lambda: before + 10.1
        self.window._frames_at = before
        with patch('live_console.time.monotonic', return_value=before + 10.1):
            self.window.poll_serial()
        self.assertEqual(self.window.machine.snapshot().state, 'recognizing')
        self.assertNotIn('STOP', self.sent)
        self.window.machine._clock = lambda: before + 31
        with patch('live_console.time.monotonic', return_value=before + 31):
            self.window.poll_serial()
        self.assertEqual(self.window.machine.snapshot().state, 'stopped')
        self.assertIn('STOP', self.sent)

    def done(self, duplicate=False):
        self.port.data = b'DONE\nDONE\n' if duplicate else b'DONE\n'
        self.window.poll_serial()

    def test_first_seen_order_color_identity_and_duplicate_done(self):
        self.start_round()
        self.recognize('yuan_zhu','蓝色')
        self.assertEqual(self.sent[-1], 'SORT,1')
        self.assertEqual(self.window.auto_bins.total, 0)
        self.assertIsNotNone(self.window.auto_bins.pending)
        self.done(duplicate=True)
        self.assertEqual(self.window.auto_bins.total, 1)
        self.recognize('yuan_zhu','蓝色')
        self.assertEqual(self.sent[-1], 'SORT,1')
        self.done()
        self.recognize('yuan_zhu','黄色')
        self.assertEqual(self.sent[-1], 'SORT,2')
        self.done()
        self.assertEqual(self.window.auto_bins.counts, (2,1,0,0,0,0))
        self.assertEqual(self.window.records.rowCount(), 3)

    def test_six_categories_then_seventh_stops_without_sort_or_binding(self):
        self.start_round()
        shapes = ('yuan_zhu','wu_leng_zhu','liu_leng_zhu','qi_leng_zhu','ba_leng_zhu','si_leng_zhu')
        for number, shape in enumerate(shapes, 1):
            self.recognize(shape,'黄色')
            self.assertEqual(self.sent[-1], 'SORT,%d' % number)
            self.done()
        # A known category still uses its box when all six are assigned.
        self.recognize('yuan_zhu','黄色')
        self.assertEqual(self.sent[-1], 'SORT,1')
        self.done()
        before = len([s for s in self.sent if s.startswith('SORT,')])
        self.recognize('yuan_huan_zhu','黄色')
        self.assertEqual(len([s for s in self.sent if s.startswith('SORT,')]), before)
        self.assertEqual(len(self.window.bindings), 6)
        self.assertEqual(self.window.machine.snapshot().state, 'stopped')
        self.assertEqual(self.window.auto_bins.total, 7)
        self.assertIsNone(self.window.auto_bins.pending)
        self.assertIn('第七', self.window.log_view.toPlainText())

    def test_manual_preview_never_allocates_and_low_confidence_never_sorts(self):
        self.recognize('yuan_zhu','蓝色',manual=True)
        self.assertEqual(self.window.bindings, {})
        self.start_round()
        before = len(self.sent)
        self.recognize('yuan_zhu','蓝色',confidence=.1)
        self.assertFalse(any(s.startswith('SORT,') for s in self.sent[before:]))
        self.assertEqual(self.window.bindings, {})

    def test_confirmed_round_restores_and_incomplete_sort_blocks_restart(self):
        self.start_round()
        self.recognize('yuan_zhu','蓝色')
        self.done()
        self.recognize('wu_leng_zhu','黄色')
        self.window.stop()
        restored = LiveWindow(data_dir=self.window.data_dir)
        self.addCleanup(restored.close)
        self.assertEqual(restored.bindings, {('yuan_zhu','蓝色'):1,('wu_leng_zhu','黄色'):2})
        self.assertEqual(restored.auto_bins.total, 1)
        self.assertIsNotNone(restored.auto_bins.pending)
        self.assertIn('1', restored.total_label.text())
        self.assertFalse(restored._round_available())

    def test_clear_round_requires_idle_and_preserves_calibration(self):
        self.start_round()
        with patch.object(QMessageBox, 'question') as question:
            self.assertFalse(self.window.clear_round())
            question.assert_not_called()
        self.window.stop()
        self.window.machine.handle_mcu('STATUS,STOP')
        self.window.auto_bins.reserve('yuan_zhu','蓝色')
        yes = getattr(QMessageBox, 'Yes', None)
        if yes is None:
            yes = QMessageBox.StandardButton.Yes
        with patch.object(QMessageBox, 'question', return_value=yes):
            self.assertTrue(self.window.clear_round())
        self.assertEqual(self.window.bindings, {})
        self.assertEqual(self.window.auto_bins.total, 0)
        self.assertIsNone(self.window.auto_bins.pending)
        self.assertTrue(self.window.calibration_valid)

    def test_bad_saved_round_is_not_silently_replaced(self):
        path = self.window.data_dir / 'live_auto_bins.json'
        path.write_text('{broken', encoding='utf8')
        before = path.read_bytes()
        restored = LiveWindow(data_dir=self.window.data_dir)
        self.addCleanup(restored.close)
        self.assertIsNotNone(restored._binding_error)
        self.assertIn('待检查', restored.total_label.text())
        self.assertEqual(path.read_bytes(), before)

    def test_reservation_disk_failure_sends_no_sort_and_allocates_nothing(self):
        self.start_round()
        before = len(self.sent)
        with patch.object(self.window.auto_bins, '_persist', side_effect=OSError('disk full')):
            self.recognize('yuan_zhu','蓝色')
        self.assertFalse(any(s.startswith('SORT,') for s in self.sent[before:]))
        self.assertEqual(self.window.bindings, {})
        self.assertEqual(self.window.machine.snapshot().state, 'stopped')

    def test_done_persistence_failure_stops_and_keeps_reservation_for_review(self):
        self.start_round()
        self.recognize('yuan_zhu','蓝色')
        with patch.object(self.window.auto_bins, '_persist', side_effect=OSError('disk full')):
            self.done()
        self.assertEqual(self.window.machine.snapshot().state, 'stopped')
        self.assertIsNotNone(self.window.auto_bins.pending)
        self.assertEqual(self.window.auto_bins.total, 0)
        self.assertFalse(self.window._round_available())
        self.assertIn('DONE', self.window.log_view.toPlainText())

    def test_directory_sync_failure_never_sends_sort_and_published_pending_survives(self):
        from auto_bins import AutoBinSession
        self.start_round()
        before = len(self.sent)
        def fail_after_publication(path):
            if self.window.auto_bins_path.exists():
                raise OSError('sync failed after publication')
        with patch('auto_bins._sync_directory', side_effect=fail_after_publication):
            self.recognize('yuan_zhu','蓝色')
        self.assertFalse(any(s.startswith('SORT,') for s in self.sent[before:]))
        restored = AutoBinSession(self.window.auto_bins_path, self.window.names, self.window.auto_bins._colors)
        self.assertIsNotNone(restored.pending)
        self.assertEqual(self.window.machine.snapshot().state, 'stopped')

    def test_offline_round_clear_requires_physical_shutdown_confirmation(self):
        self.window._finish_serial_close()
        self.assertFalse(self.window.machine.connected)
        yes = getattr(QMessageBox, 'Yes', None)
        if yes is None:
            yes = QMessageBox.StandardButton.Yes
        with patch.object(QMessageBox, 'question', return_value=yes) as question:
            self.assertTrue(self.window.clear_round())
        message = question.call_args[0][2]
        self.assertIn('供电', message)
        self.assertIn('停止', message)
        self.assertIn('清空', message)

    def test_clearing_round_waits_for_real_stop_ack_and_next_round_counts_from_one(self):
        self.start_round()
        self.recognize('yuan_zhu','蓝色')
        self.done()
        old_round = self.window.auto_bins.round_id
        self.window.stop()
        with patch.object(QMessageBox, 'question') as question:
            self.assertFalse(self.window.clear_round())
            question.assert_not_called()
        self.window.machine.handle_mcu('STATUS,STOP')
        yes = getattr(QMessageBox, 'Yes', None)
        if yes is None:
            yes = QMessageBox.StandardButton.Yes
        with patch.object(QMessageBox, 'question', return_value=yes):
            self.assertTrue(self.window.clear_round())
        self.assertNotEqual(self.window.auto_bins.round_id, old_round)
        self.window.machine.confirm_origin()
        self.window.machine.handle_mcu('READY')
        self.start_round()
        self.recognize('wu_leng_zhu','黄色')
        self.assertEqual(self.sent[-1], 'SORT,1')
        self.done()
        self.assertEqual(self.window.auto_bins.total, 1)
        self.assertEqual(self.window.records.rowCount(), 1)
        self.assertEqual(self.window.records.item(0,0).text(), '1')

    def test_inspector_is_readonly_and_shows_automatic_mapping(self):
        from live_console import QTableWidget, QAbstractItemView
        self.start_round()
        self.recognize('yuan_zhu','蓝色')
        self.done()
        dialog = self.window.create_binding_dialog()
        self.addCleanup(dialog.reject)
        table = dialog.findChild(QTableWidget, 'auto_bin_table')
        no_edit = getattr(QAbstractItemView, 'NoEditTriggers', None)
        if no_edit is None:
            no_edit = QAbstractItemView.EditTrigger.NoEditTriggers
        self.assertEqual(table.editTriggers(), no_edit)
        self.assertEqual(table.item(0,1).text(), 'yuan_zhu')
        self.assertEqual(table.item(0,2).text(), '蓝色')
        self.assertEqual(table.item(0,3).text(), '1')
