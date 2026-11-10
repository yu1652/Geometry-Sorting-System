import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from app import QApplication
from live_console import LiveWindow
from scene_presence import PresenceResult


APP = QApplication.instance() or QApplication([])


class Linux20261107Tests(unittest.TestCase):
    def test_release_uses_actual_model_order_and_distinct_engine_files(self):
        from linux_20261107 import CLASS_NAMES, make_recognizer, make_tray_backend
        self.assertEqual(CLASS_NAMES, ('wlz', 'llz', 'zft', 'slz', 'yzhu', 'yzhui', 'smt', 'qiu'))
        backend = SimpleNamespace(close=lambda: None)
        with patch('trt_backend.create_backend', return_value=backend) as create:
            recognizer = make_recognizer()
            self.assertEqual(recognizer._class_names, CLASS_NAMES)
            self.assertEqual(create.call_args.kwargs['model_kind'], 'yolo_win8')
            self.assertEqual(create.call_args.args[0].parent.name, 'linux_20261107')
            self.assertIs(make_tray_backend(), backend)
            self.assertEqual(create.call_args.kwargs['model_kind'], 'tray')
            self.assertEqual(create.call_args.args[0].parent.name, 'linux_20261107')

    def test_butu_uses_new_class_order_and_separate_round_with_same_safety_protocol(self):
        from butu_20261110 import CLASS_NAMES
        with tempfile.TemporaryDirectory() as folder:
            window = LiveWindow(data_dir=folder, butu_24=True)
            try:
                self.assertEqual(window.names, CLASS_NAMES)
                self.assertEqual(window._shape_name('ban_yuan_zhu'), '半圆柱')
                self.assertTrue(window.linux_20261107)
                self.assertEqual(window.auto_bins_path.name, 'live_auto_bins_butu24.json')
                self.assertIs(window.pages.currentWidget(), window.competition_page)
                self.assertFalse(window.start())
            finally:
                window.close()

    def test_release_opens_competition_with_debug_and_calibration_still_accessible(self):
        with tempfile.TemporaryDirectory() as folder:
            window = LiveWindow(data_dir=folder, linux_20261107=True)
            try:
                self.assertIs(window.pages.currentWidget(), window.competition_page)
                self.assertIsNotNone(window.release_competition)
                window.show_debug()
                self.assertIs(window.pages.currentWidget(), window.debug_page)
                self.assertIsNot(window.tray_log_view, window.serial_log_view)
                self.assertIsNot(window.log_view, window.serial_log_view)
                window.show_competition()
                self.assertIs(window.pages.currentWidget(), window.competition_page)
            finally:
                window.close()

    def test_hidden_competition_bin_refreshes_to_latest_confirmed_image(self):
        with tempfile.TemporaryDirectory() as folder:
            window = LiveWindow(data_dir=folder, linux_20261107=True)
            try:
                window.show()
                APP.processEvents()
                competition = window.release_competition
                image = competition.card_tiles[0][2]
                for total, bgr in ((1, (0, 0, 255)), (2, (255, 0, 0))):
                    if total == 2:
                        window.show_debug()
                    bin_frame = np.full((24, 32, 3), bgr, np.uint8)
                    competition.render(total, [('蓝色 五棱柱', total, bin_frame)] +
                                       [('待绑定', 0, None)] * 5, '待机', {})
                window.show_competition()
                APP.processEvents()
                self.assertEqual(image.pixmap().toImage().pixelColor(0, 0).name(), '#0000ff')
                competition.mode_picker.setCurrentText('表格')
                APP.processEvents()
                self.assertEqual((competition.table.rowCount(), competition.table.columnCount()), (4, 6))
                self.assertEqual(competition.table.item(3, 0).text(), '2')
                self.assertEqual(competition.table_images[0].pixmap().toImage().pixelColor(0, 0).name(), '#0000ff')
            finally:
                window.close()

    def test_butu_round_repushes_empty_then_sorts_twelve_into_six_bins(self):
        with tempfile.TemporaryDirectory() as folder:
            window = LiveWindow(data_dir=folder, butu_24=True)
            machine = window.machine
            sent = []
            machine._send = sent.append
            try:
                self.assertTrue(machine.begin_handshake())
                for reply in ('PONG', 'VERSION,ESP32_SORTER_XY_PUSHER,0.1.0',
                              'STATUS,MCU,WAIT_START,0,0,77'):
                    machine.handle_mcu(reply)
                self.assertTrue(machine.confirm_origin())
                self.assertTrue(machine.handle_mcu('READY'))
                window.save_calibration(window.calibration_rows)
                window.calibration_confirmed.setChecked(True)
                window.vision = SimpleNamespace(capture_empty=lambda _: True,
                                                locate=lambda *a: True,
                                                recognize=lambda *a, **kw: True)
                window._frames_at = time.monotonic()
                window._models_warm = True
                window.empty_reference = SimpleNamespace(assess=lambda frames, _: PresenceResult(
                    'empty' if frames is window.frames else 'occupied', 'replay', {}))
                self.assertTrue(window.start())
                self.assertTrue(machine.handle_mcu('STATUS,HOST_READY'))
                self.assertTrue(machine.handle_mcu('STATUS,START'))

                def reach_scan():
                    self.assertTrue(machine.handle_mcu('PUSH?'))
                    window._dispatch_vision()
                    window.on_located(window._job[0], (100, 75))
                    self.assertEqual(sent[-1], 'PUSH,100.000,75.000,mm')
                    self.assertTrue(machine.handle_mcu('SCAN'))

                def scan(detection, color, captures, attempts):
                    for _ in range(attempts):
                        window._scan_next_at = 0
                        window._dispatch_vision()
                        result = SimpleNamespace(detection=detection, color=color, error=None,
                            elapsed_ms=20., mask_status=('applied', 'applied'),
                            preview_bgr=np.zeros((320, 640, 3), np.uint8),
                            diagnostics={'captures': captures})
                        window.on_recognized(window._job[0], result)

                reach_scan()
                scan(None, None, window.frames, 3)
                self.assertEqual(sent[-1], 'PUSH,WAIT')
                self.assertEqual(window.auto_bins.total, 0)
                self.assertEqual(window.bindings, {})
                shapes = ('ban_yuan_zhu', 'san_leng_zhu', 'si_leng_zhu',
                          'si_leng_zhui', 'wu_leng_zhu', 'yuan_zhu')
                color = SimpleNamespace(name='蓝色', confidence=.9, valid_pixels=120)
                for item, shape in enumerate(shapes * 2):
                    window._frames_at = time.monotonic()
                    reach_scan()
                    scan(SimpleNamespace(class_name=shape, confidence=.9), color, dict(window.frames), 2)
                    self.assertEqual(sent[-1], 'SORT,%d' % (item % 6 + 1))
                    self.assertTrue(machine.handle_mcu('DONE'))
                    self.assertFalse(machine.handle_mcu('PUSH?'))
                    for check in range(3):
                        window._dispatch_vision()
                        window.on_empty_reference(window._job[0], window.frames)
                        self.assertEqual(window.auto_bins.total, item + int(check == 2))
                self.assertEqual(window.auto_bins.counts, (2, 2, 2, 2, 2, 2))
                self.assertEqual(machine.snapshot().state, 'stopped')
                self.assertFalse(machine.handle_mcu('PUSH?'))
                self.assertEqual(len([line for line in sent if line.startswith('SORT,')]), 12)
            finally:
                window.vision = None
                machine.disconnect()
                window.close()

    def exercise_drop(self, previous_count):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        window = LiveWindow(data_dir=directory.name, linux_20261107=True)
        self.addCleanup(window.close)
        self.addCleanup(setattr, window, 'vision', None)
        self.addCleanup(window._finish_serial_close)
        sent = []
        machine = window.machine
        machine._send = sent.append
        machine.begin_handshake()
        for line in ('PONG', 'VERSION,ESP32_SORTER_XY_PUSHER,0.1.0', 'STATUS,MCU,WAIT_START,0,0,77'):
            machine.handle_mcu(line)
        machine.confirm_origin()
        machine.handle_mcu('READY')
        window.save_calibration(window.calibration_rows)
        window.calibration_confirmed.setChecked(True)
        window.vision = SimpleNamespace(capture_empty=lambda token: True)
        window._frames_at = time.monotonic()
        window.empty_reference = SimpleNamespace(assess=lambda *a: PresenceResult('empty', 'test', {}))
        for _ in range(previous_count):
            window.auto_bins.complete(window.auto_bins.reserve('wlz', '蓝色'))
        self.assertFalse(window.start())
        window._models_warm = True
        self.assertTrue(window.start())
        machine.handle_mcu('STATUS,HOST_READY')
        machine.handle_mcu('STATUS,START')
        machine.handle_mcu('PUSH?')
        machine.provide_target(80, 80)
        machine.handle_mcu('SCAN')
        bin_id = window.auto_bins.reserve('wlz', '蓝色')
        window._last_sort = ('wlz', '蓝色', bin_id,
                             SimpleNamespace(preview_bgr=np.zeros((320, 640, 3), np.uint8)))
        machine.provide_sort(bin_id)
        machine.handle_mcu('DONE')
        self.assertFalse(machine.handle_mcu('PUSH?'))
        for index in range(3):
            window._dispatch_vision()
            token = window._job[0]
            window.on_empty_reference(token, window.frames)
            self.assertEqual(window.auto_bins.total, previous_count + int(index == 2))
        self.assertTrue((Path(directory.name) / 'live_auto_bins_20261107.json').is_file())
        return window

    def test_drop_holds_new_push_and_only_counts_after_three_fresh_empty_checks(self):
        window = self.exercise_drop(0)
        self.assertTrue(window.machine.handle_mcu('PUSH?'))
        self.assertEqual(window.machine.snapshot().state, 'locating')

    def test_twelfth_cleared_drop_ends_round_before_another_push(self):
        window = self.exercise_drop(11)
        self.assertEqual(window.machine.snapshot().state, 'stopped')
        self.assertFalse(window.machine.handle_mcu('PUSH?'))
