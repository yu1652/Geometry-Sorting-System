"""Exercise tray policy through real queued decoding and target selection."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from app import QApplication
from live_console import LiveWindow
from live_vision import LiveVisionWorker
import linux_20261107 as release


APP = QApplication.instance() or QApplication([])
SOURCES = {role: '/test/' + role for role in ('left', 'right', 'top')}
# Hand-checked mapping: physical X = pixel X / 4 + 25, Y = pixel Y / 4.
POINTS = [('P%d' % (j * 3 + i + 1), px, py, x, y)
          for j, (py, y) in enumerate(((80, 20), (240, 60), (400, 100)))
          for i, (px, x) in enumerate(((160, 65), (320, 105), (480, 145)))]


class StaticCapture:
    def isOpened(self):
        return True

    def read(self):
        return True, np.full((480, 640, 3), 100, np.uint8)

    def release(self):
        pass


class RawTrayBackend:
    def __init__(self, output):
        self.output = output

    def infer(self, tensor):
        return self.output.copy()

    def close(self):
        pass


def release_output(candidates):
    """Encode full-frame boxes using the release ROI's known letterbox sizes.

    The 519x479 ROI becomes 640x591 with 24 pixels of top padding.
    Candidates are (center X, center Y, width, height, confidence).
    """
    output = np.zeros((1, 5, 8400), np.float32)
    for index, (x, y, width, height, confidence) in enumerate(candidates):
        output[0, :, index] = ((x - 16) * 640 / 519, y * 591 / 479 + 24,
                               width * 640 / 519, height * 591 / 479, confidence)
    return output


class IdentityCalibration:
    def project_pixel(self, x, y):
        return float(x), float(y)

    def map_pixel(self, x, y):
        return self.project_pixel(x, y)

    def validate_physical(self, x, y):
        if not 0 <= x <= 640 or not 0 <= y <= 480:
            raise ValueError('outside test travel')


class LiveTrayPolicyTests(unittest.TestCase):
    def worker(self, output, **settings):
        worker = LiveVisionWorker(SOURCES, capture_factory=lambda path: StaticCapture(),
                                  tray_factory=lambda: RawTrayBackend(output),
                                  drain_batches=1, **settings)
        self.addCleanup(worker.shutdown)
        return worker

    def linux_window(self, frame):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / 'live_calibration.json'
        path.write_text(json.dumps({'mode': 'live', 'points': POINTS, 'frame': frame}),
                        encoding='utf8')
        # Active model pointers belong to the user's workspace, outside this test.
        with patch('side_model_store.SideModelStore.read_active', return_value=None):
            window = LiveWindow(data_dir=directory.name, linux_20261107=True)
        self.addCleanup(window.close)
        self.addCleanup(setattr, window, 'vision', None)
        self.assertTrue(window.calibration_valid)
        window._frames_at = time.monotonic()
        return window

    def test_release_manual_locate_rejects_lower_x_reflection_candidate(self):
        # Removing the release confidence override selects blank X=86 instead.
        window = self.linux_window({'box': [200, 100, 300, 250], 'inset_percent': 18})
        output = release_output(((220, 200, 40, 40, .666), (360, 200, 40, 40, .844)))
        worker = window.vision = self.worker(output, tray_feature_input=True,
                                            tray_compensation=release.TRAY_COMPENSATION)
        results, failures, sent = [], [], []
        window.machine._send = sent.append
        self.assertTrue(window.machine.begin_handshake())
        for reply in ('PONG', 'VERSION,ESP32_SORTER_XY_PUSHER,0.1.0',
                      'STATUS,MCU,WAIT_START,0,0,77'):
            window.machine.handle_mcu(reply)
        self.assertTrue(window.machine.confirm_origin())
        self.assertTrue(window.machine.handle_mcu('READY'))
        self.assertEqual(window.machine.snapshot().state, 'armed')
        sent[:] = []
        worker.location_ready.connect(window.on_located)
        worker.location_ready.connect(lambda token, target: (results.append(target), worker.shutdown()))
        worker.failed.connect(lambda token, message: (failures.append(message), worker.shutdown()))
        self.assertTrue(window.manual_locate())
        worker.run()
        self.assertEqual(failures, [])
        np.testing.assert_allclose(results, [(121, 52.25)], atol=1e-4)
        self.assertIsNone(window._job)
        self.assertEqual(window.machine.snapshot().state, 'armed')
        self.assertEqual(sent, [])

    def test_release_uses_saved_frame_then_updated_frame_without_camera_restart(self):
        # Dropping the frame gate selects X=71 even in the first saved frame.
        initial = {'box': [200, 100, 300, 250], 'inset_percent': 18}
        updated = {'box': [150, 100, 100, 250], 'inset_percent': 18}
        window = self.linux_window(initial)
        output = release_output(((170, 200, 20, 20, .860), (220, 200, 40, 40, .359),
                                 (360, 200, 40, 40, .836)))
        worker = window.vision = self.worker(output, tray_feature_input=True,
                                            tray_compensation=release.TRAY_COMPENSATION)
        results, failures, sent = [], [], []
        window.machine._send = sent.append

        def located(token, target):
            window.on_located(token, target)
            results.append(target)
            if len(results) == 1:
                window.save_calibration(window.calibration_rows, frame=updated)
                self.assertTrue(window.manual_locate())
            else:
                worker.shutdown()

        worker.location_ready.connect(located)
        worker.failed.connect(lambda token, message: (failures.append(message), worker.shutdown()))
        self.assertTrue(window.manual_locate())
        worker.run()
        self.assertEqual(failures, [])
        np.testing.assert_allclose(results, [(121, 52.25), (71, 52.25)], atol=1e-4)
        self.assertIs(window.vision, worker)
        self.assertEqual(sent, [])
        saved = json.loads(window.calibration_path.read_text(encoding='utf8'))
        self.assertEqual(saved['points'], [list(point) for point in POINTS])
        self.assertEqual(saved['frame'], updated)

    def test_worker_custom_low_confidence_reaches_both_decoder_and_locator(self):
        # Either component retaining .35 discards this otherwise valid .25 box.
        output = np.zeros((1, 5, 8400), np.float32)
        output[0, :, 0] = (100, 280, 40, 40, .25)  # full-frame Y=200; pad=80
        worker = self.worker(output)
        results, failures = [], []
        worker.location_ready.connect(lambda token, target: (results.append(target), worker.shutdown()))
        worker.failed.connect(lambda token, message: (failures.append(message), worker.shutdown()))
        try:
            accepted = worker.locate(7, IdentityCalibration(), (0, 0, 640, 480), (4, 0),
                                     confidence=.2, selection_region=(50, 100, 150, 300))
        except TypeError as error:
            self.fail('Queued locate must accept the requested tray policy: ' + str(error))
        self.assertTrue(accepted)
        worker.run()
        self.assertEqual(failures, [])
        self.assertEqual(results, [(124, 200)])

    def test_worker_legacy_four_positional_arguments_keep_default_selection(self):
        output = np.zeros((1, 5, 8400), np.float32)
        output[0, :, 0] = (100, 280, 40, 40, .359)
        output[0, :, 1] = (300, 280, 40, 40, .836)
        worker = self.worker(output)
        results, failures = [], []
        worker.location_ready.connect(lambda token, target: (results.append(target), worker.shutdown()))
        worker.failed.connect(lambda token, message: (failures.append(message), worker.shutdown()))
        self.assertTrue(worker.locate(8, IdentityCalibration(), (0, 0, 640, 480), (4, 0)))
        worker.run()
        self.assertEqual(failures, [])
        self.assertEqual(results, [(124, 200)])

    def test_generic_window_keeps_legacy_worker_api(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        with patch('side_model_store.SideModelStore.read_active', return_value=None):
            window = LiveWindow(data_dir=directory.name)
        self.addCleanup(window.close)
        self.addCleanup(setattr, window, 'vision', None)
        window.save_calibration(window.calibration_rows)
        window._frames_at = time.monotonic()

        class LegacyWorker:
            def locate(self, token, calibration, roi, offsets):
                return True

        window.vision = LegacyWorker()
        self.assertTrue(window.manual_locate())

