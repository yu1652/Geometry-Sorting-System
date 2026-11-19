"""Real saved SCAN frames after one-part pushes, replayed without hardware."""
import json
from pathlib import Path
import unittest

import numpy as np

import linux_20261107 as release
from tray_locator import TrayLocator
from xy_calibration import CalibrationPoint, XYCalibration


class RecordedDetector:
    def __init__(self, detections):
        self.detections = detections

    def detect(self, frame):
        return self.detections


class TrayReflectionRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).parent / 'fixtures/tray_20261116_push_trial.json'
        cls.data = json.loads(path.read_text(encoding='utf-8'))
        cls.calibration = XYCalibration.fit(
            [CalibrationPoint(*row) for row in cls.data['calibration_points']],
            pixel_region=(16, 0, 535, 479),
            physical_region=cls.data['physical_region'])
        cls.frame = np.zeros(cls.data['frame_shape'], np.uint8)

    def recorded_target(self, token):
        record = next(frame for frame in self.data['frames'] if frame['token'] == token)
        locator = TrayLocator(
            self.calibration, RecordedDetector(record['detections_roi_xyxy_score']),
            release.TRAY_ROI, offsets=release.PUSH_OFFSET,
            confidence=release.TRAY_CONFIDENCE,
            compensation=release.TRAY_COMPENSATION, skip_unsafe=True,
            selection_region=self.data['selection_region_xyxy'])
        return record, locator.locate(self.frame)

    def test_green_part_remains_locatable_after_first_missed_push(self):
        record, target = self.recorded_target(2)
        np.testing.assert_allclose(target, record['expected_target_mm'], rtol=0, atol=1e-7)

    def test_green_part_remains_locatable_after_second_missed_push(self):
        record, target = self.recorded_target(5)
        np.testing.assert_allclose(target, record['expected_target_mm'], rtol=0, atol=1e-7)

    def test_reflection_does_not_request_push_after_green_part_leaves_tray(self):
        _, target = self.recorded_target(8)
        self.assertIsNone(target)

    def test_later_empty_tray_reflection_does_not_request_push(self):
        _, target = self.recorded_target(14)
        self.assertIsNone(target)


if __name__ == '__main__':
    unittest.main()
