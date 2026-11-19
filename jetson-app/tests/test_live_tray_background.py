"""Optional tray reference capture and fresh-frame locating never command motion."""
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import cv2
from app import QApplication
from live_console import LiveWindow
from live_vision import LiveVisionWorker
from tray_background import TrayBackgroundReference
import linux_20261107 as release
from motion_limits import MACHINE_REGION
from tray_locator import TrayLocator

APP = QApplication.instance() or QApplication([])
SOURCES = {role: '/test/tray-background/' + role for role in ('left', 'right', 'top')}
POINTS = [('P%d' % (j * 3 + i + 1), px, py, x, y)
          for j, (py, y) in enumerate(((80, 20), (240, 60), (400, 100)))
          for i, (px, x) in enumerate(((160, 65), (320, 105), (480, 145)))]


class Capture:
    def __init__(self):
        self.count = 0
        self.last = None

    def isOpened(self):
        return True

    def read(self):
        self.count += 1
        self.last = np.full((480, 640, 3), self.count, np.uint8)
        return True, self.last

    def release(self):
        pass


class Calibration:
    def project_pixel(self, x, y):
        return x / 4., y / 4.

    def map_pixel(self, x, y):
        return self.project_pixel(x, y)

    def validate_physical(self, x, y):
        if not 0 <= x <= 220 or not 0 <= y <= 220:
            raise ValueError('outside')


class TrayWorkerTests(unittest.TestCase):
    def worker(self, stop_on_failure=True):
        captures = {}
        def factory(source):
            captures[source] = Capture()
            return captures[source]
        worker = LiveVisionWorker(SOURCES, capture_factory=factory, drain_batches=3,
                                  tray_factory=lambda: self.fail('Reference jobs must not load YOLO'),
                                  recognizer_factory=lambda: SimpleNamespace(
                                      recognize_pair=lambda *args: SimpleNamespace(error=None), close=lambda: None))
        failures = []
        def failed(token, message):
            failures.append((token, message))
            if stop_on_failure:
                worker.shutdown()
        worker.failed.connect(failed)
        return worker, captures, failures

    def test_capture_is_independent_fresh_copy_and_fifo_before_recognition(self):
        worker, captures, failures = self.worker(stop_on_failure=False)
        self.assertTrue(callable(getattr(worker, 'capture_tray_background', None)))
        events, references, empty = [], [], []
        worker.empty_reference_ready.connect(lambda *args: empty.append(args))
        def captured(token, frame):
            events.append(('tray', token))
            references.append(frame)
            self.assertEqual(int(frame[0, 0, 0]), 7)  # 3 startup + 1 preflight + 3 drain.
            self.assertFalse(np.shares_memory(frame, captures[SOURCES['top']].last))
            self.assertTrue(worker.recognize(12))
        worker.tray_background_ready.connect(captured)
        worker.recognition_ready.connect(lambda token, result: (events.append(('recognize', token)), worker.shutdown()))
        self.assertTrue(worker.capture_tray_background(11))
        self.assertFalse(worker.recognize(99))
        worker.run()
        self.assertEqual(events, [('tray', 11), ('recognize', 12)])
        self.assertEqual(empty, [])
        self.assertEqual(failures[0][0], 99)
        self.assertEqual(int(references[0][0, 0, 0]), 7)

    def test_foreground_primary_evidence_precedes_target_with_exact_fresh_top(self):
        worker, captures, failures = self.worker()
        roi = (16, 0, 200, 200)
        seen, events = [], []
        class Detector:
            def __init__(self, reference):
                np.testing.assert_array_equal(reference, np.zeros((200, 200, 3), np.uint8))
                self.last_diagnostics = {'foreground_mask': np.ones((200, 200), np.uint8), 'candidates': []}
            def detect(self, frame):
                seen.append(frame.copy())
                return [(80, 80, 100, 100, 1.0)]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'reference.npz'
            TrayBackgroundReference(np.zeros((480, 640, 3), np.uint8), SOURCES['top'], roi).save(path)
            class MutablePath:
                value = str(path)
                def __str__(self):
                    return self.value
            queued_path = MutablePath()
            worker.location_evidence_ready.connect(lambda token, evidence: events.append(('evidence', token, evidence)))
            worker.location_ready.connect(lambda token, target: (events.append(('target', token, target)), worker.shutdown()))
            with patch.dict('sys.modules', {'tray_foreground': SimpleNamespace(TrayForegroundDetector=Detector)}):
                self.assertTrue(worker.locate(7, Calibration(), roi, (4, 0),
                                             background_reference=queued_path, motion_target_only=True))
                queued_path.value = str(path.with_name('not-the-queued-reference.npz'))
                worker.run()
        self.assertEqual(failures, [])
        self.assertEqual([event[0] for event in events], ['evidence', 'target'])
        evidence = events[0][2]
        np.testing.assert_array_equal(evidence['captures']['top'], captures[SOURCES['top']].last)
        self.assertEqual(int(seen[0][0, 0, 0]), 7)
        self.assertEqual(evidence['source_top'], SOURCES['top'])
        self.assertEqual(evidence['detector_kind'], 'foreground')

    def test_reference_camera_mismatch_fails_token_without_yolo_fallback(self):
        worker, _, failures = self.worker()
        roi = (16, 0, 200, 200)
        targets = []
        worker.location_ready.connect(lambda *args: targets.append(args))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'reference.npz'
            TrayBackgroundReference(np.zeros((480, 640, 3), np.uint8), '/different/top', roi).save(path)
            self.assertTrue(worker.locate(8, Calibration(), roi, (4, 0), background_reference=path))
            worker.run()
        self.assertEqual(targets, [])
        self.assertEqual(failures[0][0], 8)
        self.assertIn('camera', failures[0][1].lower())

    def test_reference_roi_mismatch_fails_without_yolo_fallback(self):
        worker, _, failures = self.worker()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'reference.npz'
            TrayBackgroundReference(np.zeros((480, 640, 3), np.uint8), SOURCES['top'], (16, 0, 190, 200)).save(path)
            self.assertTrue(worker.locate(9, Calibration(), (16, 0, 200, 200), (4, 0), background_reference=path))
            worker.run()
        self.assertEqual(failures[0][0], 9)
        self.assertIn('roi', failures[0][1].lower())


class TrayWindowTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        with patch('side_model_store.SideModelStore.read_active', return_value=None):
            self.window = LiveWindow(data_dir=self.folder.name, linux_20261107=True)
        self.window.timer.stop()
        self.addCleanup(self.window.close)
        self.addCleanup(setattr, self.window, 'vision', None)
        self.sent, self.requests = [], []
        self.window.machine._send = self.sent.append
        self.window.vision = SimpleNamespace(capture_tray_background=lambda token: self.requests.append(token) or True)
        self.window._frames_at = time.monotonic()

    def test_explicit_capture_saves_only_matching_top_without_motion(self):
        machine = self.window.machine
        machine._connected = machine.ready = machine._origin = True
        machine._state, machine._mcu_state = 'armed', 'WAIT_START'
        self.assertTrue(self.window.save_tray_background())
        token = self.requests[0]
        frame = np.full((480, 640, 3), 31, np.uint8)
        self.window.on_tray_background_ready(token, frame)
        loaded = TrayBackgroundReference.load(self.window.tray_background_path,
            expected_source_top=self.window.sources['top'], expected_roi=release.TRAY_FOREGROUND_ROI)
        np.testing.assert_array_equal(loaded.frame, frame)
        self.assertIsNone(self.window._job)
        self.assertEqual(self.sent, [])

    def test_stale_epoch_or_changed_camera_cannot_overwrite_reference(self):
        for change in ('epoch', 'source'):
            self.assertTrue(self.window.save_tray_background())
            token = self.requests[-1]
            if change == 'epoch':
                self.window.machine.epoch += 1
            else:
                self.window.sources['top'] += '-changed'
            self.window.on_tray_background_ready(token, np.zeros((480, 640, 3), np.uint8))
            self.assertFalse(self.window.tray_background_path.exists())
        self.assertEqual(self.sent, [])

    def test_running_machine_cannot_capture_reference(self):
        self.window.machine._state = 'running'
        self.assertFalse(self.window.save_tray_background())
        self.assertEqual(self.requests, [])
        self.assertEqual(self.sent, [])

    def test_changed_settings_require_reconnecting_the_actual_top_camera(self):
        self.window.vision.sources = dict(self.window.sources, top='/previous/physical-camera')
        self.assertFalse(self.window.save_tray_background())
        self.assertEqual(self.requests, [])

    def test_motion_started_during_capture_cannot_save_reference(self):
        self.assertTrue(self.window.save_tray_background())
        self.window.machine._pending = 'goto'
        self.window.on_tray_background_ready(self.requests[-1], np.zeros((480, 640, 3), np.uint8))
        self.assertFalse(self.window.tray_background_path.exists())
        self.assertEqual(self.sent, [])

    def test_reference_removes_blue_crop_and_uses_machine_target_validation(self):
        self.window.save_calibration(POINTS)
        TrayBackgroundReference(np.zeros((480, 640, 3), np.uint8), self.window.sources['top'], release.TRAY_FOREGROUND_ROI).save(
            self.window.tray_background_path)
        self.window.vision.locate = lambda *args, **kwargs: self.requests.append((args, kwargs)) or True
        self.assertTrue(self.window.manual_locate())
        args, kwargs = self.requests[-1]
        self.assertEqual(kwargs['background_reference'], str(self.window.tray_background_path))
        self.assertIsNone(kwargs['selection_region'])
        self.assertTrue(kwargs['motion_target_only'])
        self.assertEqual(tuple(args[1].physical_region), MACHINE_REGION)
        self.assertEqual(args[2], release.TRAY_FOREGROUND_ROI)
        self.assertEqual(tuple(args[1].pixel_region), (0, 0, 640, 480))
        self.assertEqual(self.sent, [])

    def test_tray_evidence_is_separate_and_stale_token_does_not_save(self):
        self.window._job = (5, self.window.machine.epoch, 'manual_locate')
        evidence = {'captures': {'top': np.zeros((480, 640, 3), np.uint8)},
                    'locator': {'hold_reason': 'outside machine'}, 'detector': {}, 'source_top': 'top'}
        with patch.object(self.window.diagnostics, 'save_sample', return_value=Path('tray.zip')) as save:
            self.window.on_location_evidence(4, evidence)
            self.assertFalse(save.called)
            self.window.on_location_evidence(5, evidence)
            self.assertEqual(save.call_args[1]['group'], 'tray')
            self.assertEqual(self.window._job[0], 5)
            self.window.on_located(5, None)
        self.assertEqual('outside machine', self.window._tray_status)
        self.assertEqual(self.sent, [])

    def test_camera_and_start_never_automatically_capture_background(self):
        class Signal:
            def connect(self, callback):
                pass
        worker = SimpleNamespace(start=lambda: None)
        for name in ('frames_ready', 'recognition_ready', 'location_ready', 'location_evidence_ready',
                     'tray_background_ready', 'failed', 'notice', 'models_ready', 'finished'):
            setattr(worker, name, Signal())
        worker.capture_tray_background = lambda *args: self.fail('Startup must not capture the tray')
        worker.capture_empty = lambda *args: self.fail('Startup must not capture side emptiness')
        self.window.vision = None
        self.window._vision_factory = lambda *args, **kwargs: worker
        self.window.connect_cameras()
        self.assertIs(self.window.vision, worker)
        self.window.machine._origin = True
        with patch.object(self.window, '_start_ready', return_value=True), \
             patch.object(self.window, '_ask_origin_unchanged', return_value=True), \
             patch.object(self.window.machine, 'start', return_value=True):
            self.assertTrue(self.window.start())

    def test_blocked_earlier_piece_only_waits_and_does_not_count_or_sort(self):
        machine = self.window.machine
        machine._connected = machine.ready = machine._origin = True
        machine._state = 'locating'
        self.window._job = (8, machine.epoch, 'locate')
        evidence = {'locator': {'hold_reason': 'negative Y target'}, 'detector': {}}
        with patch.object(self.window, '_automatic_ready', return_value=True), \
             patch.object(self.window.diagnostics, 'save_sample', return_value=Path('tray.zip')):
            self.window.on_location_evidence(8, evidence)
            self.window.on_located(8, None)
        self.assertEqual(self.sent, ['PUSH,WAIT'])
        self.assertEqual('negative Y target', self.window._tray_status)
        self.assertEqual(machine.snapshot().completed_count, 0)
        self.assertEqual(self.window.auto_bins.total, 0)
        self.assertIsNone(self.window.auto_bins.pending)
        machine.disconnect()

    def test_next_locate_clears_previous_hold_reason_even_without_evidence(self):
        self.window.save_calibration(POINTS)
        self.window.vision.locate = lambda *args, **kwargs: True
        self.window._tray_location_hold = (2, self.window.machine.epoch, 'old blocked piece')
        self.assertTrue(self.window.manual_locate())
        self.assertIsNone(self.window._tray_location_hold)
        self.window.on_located(self.window._job[0], None)
        self.assertEqual(self.window._tray_status, '无目标')

    def overlay_evidence(self, blocked=False):
        class Detector:
            def detect(self, frame):
                return [(200, 100, 240, 140, 1.), (40, 60, 80, 100, 1.)]
        class Bounds(Calibration):
            def validate_physical(self, x, y):
                super().validate_physical(x, y)
                if blocked and x < 30:
                    raise ValueError('unreachable first part')
        locator = TrayLocator(Bounds(), Detector(), (0, 0, 640, 480), (4, 0),
                              column_boundaries=(95, 152), motion_target_only=True)
        locator.locate(np.zeros((480, 640, 3), np.uint8))
        return {'locator': locator.last_diagnostics, 'source_top': self.window.sources['top'], 'detector': {}}

    def cache_overlay(self, blocked=False):
        self.window._job = (51, self.window.machine.epoch, 'manual_locate')
        with patch('live_console.time.monotonic', return_value=100.), \
             patch.object(self.window.diagnostics, 'save_sample', return_value=Path('tray.zip')):
            self.window.on_location_evidence(51, self.overlay_evidence(blocked))

    def test_real_trace_overlay_labels_follow_selection_keys_and_preserve_raw_frames(self):
        self.window.show_debug()
        self.cache_overlay()
        frames = {role: np.zeros((480, 640, 3), np.uint8) for role in SOURCES}
        with patch('live_console.time.monotonic', return_value=100.1), \
             patch('live_console.cv2.putText', wraps=cv2.putText) as text, \
             patch.object(self.window.views['top'], 'show_frame') as show:
            self.window.on_frames(frames)
        rendered = show.call_args[0][0]
        np.testing.assert_array_equal(rendered[60, 40], (0, 255, 0))
        np.testing.assert_array_equal(rendered[100, 200], (0, 255, 255))
        labels = {call[0][1]: call[0][2] for call in text.call_args_list if call[0][1].startswith('#')}
        self.assertEqual(labels, {'#1': (40, 56), '#2': (200, 96)})
        for role in frames:
            self.assertIs(self.window.frames[role], frames[role])
            self.assertFalse(np.any(frames[role]))
        self.assertEqual(self.sent, [])

    def test_real_blocked_trace_is_red_and_competition_view_uses_same_overlay(self):
        self.cache_overlay(blocked=True)
        frames = {role: np.zeros((480, 640, 3), np.uint8) for role in SOURCES}
        self.window._competition_preview_at = 0
        with patch('live_console.time.monotonic', return_value=100.1), \
             patch.object(self.window.release_competition, 'set_frames') as show:
            self.window.on_frames(frames)
        rendered = show.call_args[0][1]
        np.testing.assert_array_equal(rendered[60, 40], (0, 0, 255))
        np.testing.assert_array_equal(rendered[100, 200], (0, 255, 255))
        self.assertFalse(np.any(frames['top']))
        self.assertEqual(self.sent, [])

    def test_expired_changed_source_or_epoch_clears_overlay(self):
        for change in ('expired', 'source', 'epoch'):
            self.cache_overlay()
            if change == 'source':
                self.window.sources['top'] += '-new'
            elif change == 'epoch':
                self.window.machine.epoch += 1
            with patch('live_console.time.monotonic', return_value=102. if change == 'expired' else 100.1):
                rendered = self.window._tray_candidate_preview(np.zeros((480, 640, 3), np.uint8))
            self.assertFalse(np.any(rendered))
            self.assertIsNone(self.window._tray_location_overlay)
        self.window._job = (52, self.window.machine.epoch - 1, 'manual_locate')
        self.window.on_location_evidence(52, self.overlay_evidence())
        self.assertIsNone(self.window._tray_location_overlay)


if __name__ == '__main__':
    unittest.main()
