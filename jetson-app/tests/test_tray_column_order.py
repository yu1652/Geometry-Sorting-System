"""Column-first push order through geometry and the real queued locate path."""
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
from tray_locator import TrayLocator
from xy_calibration import XYCalibration


APP = QApplication.instance() or QApplication([])
SOURCES = {role: '/column-order-test/' + role for role in ('left', 'right', 'top')}
POINTS = [('P%d' % (j * 3 + i + 1), px, py, x, y)
          for j, (py, y) in enumerate(((80, 20), (240, 60), (400, 100)))
          for i, (px, x) in enumerate(((160, 65), (320, 105), (480, 145)))]


class RecordedDetector:
    def __init__(self, detections):
        self.detections = detections

    def detect(self, frame):
        return self.detections


def identity_calibration(physical_region=(0, 0, 220, 220)):
    # Exact X=px,Y=py coefficients avoid fit noise in sort-boundary checks.
    return XYCalibration(np.array(((0, 0), (1, 0), (0, 1), (0, 0), (0, 0), (0, 0))),
                         (0, 0, 220, 220), physical_region)


def box(x, y, score=.9):
    return (x - 2, y - 2, x + 2, y + 2, score)


class TrayColumnOrderTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((220, 220, 3), np.uint8)

    def locator(self, detections, **settings):
        calibration = settings.pop('calibration', identity_calibration())
        settings.setdefault('column_boundaries', (95, 152))
        try:
            return TrayLocator(calibration, RecordedDetector(detections), (0, 0, 220, 220),
                               offsets=(1, 0), **settings)
        except TypeError as error:
            self.fail('Locate must accept the explicit column-order policy: ' + str(error))

    def test_shuffled_nine_parts_follow_p1_p4_p7_then_p2_p5_p8_then_p3_p6_p9(self):
        # Real field X offsets within a column must not put P4/P7 before P1.
        pieces = {'P1': box(67, 10), 'P2': box(123, 10), 'P3': box(180, 10),
                  'P4': box(66, 58), 'P5': box(124, 58), 'P6': box(180, 58),
                  'P7': box(66, 115), 'P8': box(124.2, 115), 'P9': box(180, 115)}
        remaining = [pieces['P%d' % n] for n in range(9, 0, -1)]
        locator = self.locator(remaining, skip_unsafe=True)
        for name, expected in (('P1', (70, 10)), ('P4', (69, 58)), ('P7', (69, 115)),
                               ('P2', (126, 10)), ('P5', (127, 58)), ('P8', (127.2, 115)),
                               ('P3', (183, 10)), ('P6', (183, 58)), ('P9', (183, 115))):
            with self.subTest(point=name):
                np.testing.assert_allclose(locator.locate(self.frame), expected, rtol=0, atol=1e-9)
            remaining.remove(pieces[name])
        self.assertIsNone(locator.locate(self.frame))

    def test_same_column_uses_y_before_small_x_differences(self):
        result = self.locator([box(66, 115), box(67, 10), box(66, 58)]).locate(self.frame)
        self.assertEqual(result, (70, 10))

    def test_earlier_column_finishes_before_smaller_y_in_later_column(self):
        result = self.locator([box(110, 10), box(80, 180)]).locate(self.frame)
        self.assertEqual(result, (83, 180))

    def test_equal_y_within_column_uses_smaller_x(self):
        result = self.locator([box(90, 50), box(70, 50)]).locate(self.frame)
        self.assertEqual(result, (73, 50))

    def test_boundary_center_belongs_to_column_on_its_right(self):
        for edge, left_x in ((95, 94), (152, 151)):
            with self.subTest(edge=edge):
                result = self.locator([box(edge, 10), box(left_x, 100)]).locate(self.frame)
                self.assertEqual(result, (left_x + 3, 100))

    def test_omitted_column_policy_preserves_strict_smallest_x_default(self):
        result = TrayLocator(identity_calibration(),
                             RecordedDetector([box(67, 10), box(66, 115)]),
                             (0, 0, 220, 220), offsets=(1, 0)).locate(self.frame)
        self.assertEqual(result, (69, 115))

    def test_invalid_column_boundaries_are_rejected(self):
        for boundaries in ((), (95,), (95, 152, 180), (95, 95), (152, 95),
                           (float('nan'), 152), (95, float('inf')), 'bad', ((95,), (152,))):
            with self.subTest(boundaries=boundaries):
                with self.assertRaises(ValueError):
                    self.locator([], column_boundaries=boundaries)

    def test_order_does_not_restore_low_confidence_outside_frame_or_unsafe_candidates(self):
        boxes = [box(40, 10, .70), box(55, 15), box(65, 20), box(199, 25), box(100, 80)]
        locator = self.locator(boxes, confidence=.75, selection_region=(60, 0, 220, 220),
                               calibration=identity_calibration((70, 0, 200, 220)),
                               skip_unsafe=True)
        # X40 fails confidence, X55 the frame, X65 the physical center, and
        # X199 the rear+gap target. Only X100 is eligible for sorting.
        self.assertEqual(locator.locate(self.frame), (103, 80))


class StaticCapture:
    def isOpened(self):
        return True

    def read(self):
        return True, np.zeros((480, 640, 3), np.uint8)

    def release(self):
        pass


class RawBackend:
    def infer(self, tensor):
        output = np.zeros((1, 5, 8400), np.float32)
        # Full-frame centers A=(188,120), B=(180,280), box width/height=8.
        # Under POINTS: A=(72,30)mm, B=(70,70)mm, both in the first column.
        for i, (x, y) in enumerate(((188, 120), (180, 280))):
            output[0, :, i] = ((x - 16) * 640 / 519, y * 591 / 479 + 24,
                               8 * 640 / 519, 8 * 591 / 479, .9)
        return output

    def close(self):
        pass


class LiveColumnOrderTests(unittest.TestCase):
    def worker(self):
        worker = LiveVisionWorker(SOURCES, capture_factory=lambda _: StaticCapture(),
                                  tray_factory=RawBackend, drain_batches=1,
                                  tray_compensation=release.TRAY_COMPENSATION)
        self.addCleanup(worker.shutdown)
        return worker

    def window(self, rows=POINTS):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        (Path(directory.name) / 'live_calibration.json').write_text(
            json.dumps({'mode': 'live', 'points': rows,
                        'frame': {'box': [150, 20, 360, 400], 'inset_percent': 15}}), encoding='utf8')
        with patch('side_model_store.SideModelStore.read_active', return_value=None):
            window = LiveWindow(data_dir=directory.name, linux_20261107=True)
        self.addCleanup(window.close)
        self.addCleanup(setattr, window, 'vision', None)
        window._frames_at = time.monotonic()
        self.assertTrue(window.calibration_valid)
        return window

    def test_explicit_column_policy_survives_real_worker_queue_and_caller_mutation(self):
        from xy_calibration import CalibrationPoint
        calibration = XYCalibration.fit([CalibrationPoint(*row[1:]) for row in POINTS],
                                        pixel_region=(16, 0, 535, 479),
                                        physical_region=release.PICK_REGION)
        worker, targets, errors = self.worker(), [], []
        worker.location_ready.connect(lambda token, target: (targets.append(target), worker.shutdown()))
        worker.failed.connect(lambda token, message: (errors.append(message), worker.shutdown()))
        boundaries = [85, 125]
        try:
            accepted = worker.locate(1, calibration, release.TRAY_ROI, release.PUSH_OFFSET,
                                     confidence=release.TRAY_CONFIDENCE, column_boundaries=boundaries)
        except TypeError as error:
            self.fail('Worker must queue the column-order policy: ' + str(error))
        self.assertTrue(accepted)
        boundaries[0] = 71  # A retained reference would incorrectly place B in an earlier column.
        worker.run()
        self.assertEqual(errors, [])
        np.testing.assert_allclose(targets, [(74, 33.75)], rtol=0, atol=1e-4)

    def test_linux_uses_named_column_medians_and_refreshes_after_saving_without_camera_restart(self):
        # Shuffle rows to ensure P names, rather than table order, define columns.
        window = self.window(POINTS[4:] + POINTS[:4])
        worker = window.vision = self.worker()
        targets, errors, sent = [], [], []
        window.machine._send = sent.append

        def located(token, target):
            window.on_located(token, target)
            targets.append(target)
            if len(targets) == 1:
                # New anchors=78,126,174 => boundaries=102,150.
                # Cached old boundary85 would put B(84) before A(86.4).
                rows = [(name, px, py, mx * 1.2, my) for name, px, py, mx, my in POINTS]
                window.save_calibration(rows)
                self.assertTrue(window.manual_locate())
            else:
                worker.shutdown()

        worker.location_ready.connect(located)
        worker.failed.connect(lambda token, message: (errors.append(message), worker.shutdown()))
        self.assertTrue(window.manual_locate())
        worker.run()
        self.assertEqual(errors, [])
        np.testing.assert_allclose(targets, [(74, 33.75), (88.6, 33.75)], rtol=0, atol=1e-4)
        self.assertIs(window.vision, worker)
        self.assertEqual(sent, [])

    def test_nonincreasing_named_columns_reject_locate_before_any_motion(self):
        rows = [(name, px, py, 210 - mx, my) for name, px, py, mx, my in POINTS]
        window = self.window(rows)
        window.vision = self.worker()
        sent = []
        window.machine._send = sent.append
        self.assertFalse(window.manual_locate())
        self.assertIsNone(window._job)
        self.assertEqual(sent, [])


if __name__ == '__main__':
    unittest.main()
