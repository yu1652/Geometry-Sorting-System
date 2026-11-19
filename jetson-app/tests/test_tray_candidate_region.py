import json
import os
import unittest

import numpy as np

from tray_locator import TrayLocator
from xy_calibration import CalibrationPoint, XYCalibration


class RecordedDetector:
    """Supply decoded observations without requiring a camera or TensorRT."""

    def __init__(self, detections):
        self.detections = detections

    def detect(self, frame):
        return self.detections


class RecordedTrayCandidateTests(unittest.TestCase):
    def setUp(self):
        path = os.path.join(os.path.dirname(__file__), 'fixtures',
                            'tray_20261116_candidates.json')
        with open(path, encoding='utf-8') as handle:
            self.fixture = json.load(handle)
        data = self.fixture
        self.calibration = XYCalibration.fit(
            [CalibrationPoint(*row) for row in data['calibration_points']],
            pixel_region=(16, 0, 535, 479),
            physical_region=data['physical_region'])
        self.frame = np.zeros(data['frame_shape'], dtype=np.uint8)

    def locator(self, detections, **settings):
        data = self.fixture
        return TrayLocator(self.calibration, RecordedDetector(detections), data['roi_xywh'],
                           offsets=data['offsets_mm'], compensation=data['compensation'],
                           skip_unsafe=True, **settings)

    def test_recorded_green_object_wins_with_release_threshold_and_frame(self):
        # The frame extends beyond the detector ROI and the object's box crosses
        # its bottom edge. Only the full-frame box center grants eligibility.
        locator = self.locator(self.fixture['detections_roi_xyxy_score'], confidence=.5,
                               selection_region=self.fixture['selection_region_xyxy'])
        np.testing.assert_allclose(locator.locate(self.frame),
                                   self.fixture['recorded_green_target_mm'], atol=1e-7, rtol=0)

    def test_absent_green_leaves_no_eligible_recorded_candidate(self):
        # At .5, candidate 5 still passes physical checks but its center is left
        # of the user frame. Candidate 11 is inside the frame but below .5.
        locator = self.locator(self.fixture['detections_roi_xyxy_score'][1:], confidence=.5,
                               selection_region=self.fixture['selection_region_xyxy'])
        self.assertIsNone(locator.locate(self.frame))

    def test_omitted_region_and_default_confidence_preserve_legacy_selection(self):
        locator = self.locator(self.fixture['detections_roi_xyxy_score'])
        np.testing.assert_allclose(locator.locate(self.frame),
                                   self.fixture['recorded_legacy_blank_target_mm'], atol=1e-7, rtol=0)

    def test_none_region_retains_physically_safe_extrapolated_candidate(self):
        locator = self.locator(self.fixture['detections_roi_xyxy_score'][1:],
                               confidence=.5, selection_region=None)
        np.testing.assert_allclose(locator.locate(self.frame),
                                   self.fixture['recorded_outside_frame_target_mm'], atol=1e-7, rtol=0)


class TraySelectionRegionTests(unittest.TestCase):
    roi = (10, 20, 80, 80)
    region = (30, 40, 70, 80)

    def setUp(self):
        self.frame = np.zeros((120, 120, 3), dtype=np.uint8)

    def calibration(self, reverse_x=False, physical_region=(0, 0, 200, 200)):
        # Exact affine coefficients keep boundary expectations independent of
        # polynomial fit rounding: X=x (or 120-x), Y=y.
        coefficients = np.array([[120 if reverse_x else 0, 0],
                                 [-1 if reverse_x else 1, 0], [0, 1],
                                 [0, 0], [0, 0], [0, 0]], dtype=float)
        return XYCalibration(coefficients, (0, 0, 200, 200), physical_region)

    def locator(self, detections=(), **settings):
        calibration = settings.pop('calibration', self.calibration())
        settings.setdefault('selection_region', self.region)
        return TrayLocator(calibration, RecordedDetector(detections), self.roi, **settings)

    def test_all_four_full_frame_center_boundaries_are_inclusive(self):
        for box, target in (
            ((15, 35, 25, 45, .9), (39, 60)),  # Left, with corners outside frame.
            ((55, 35, 65, 45, .9), (79, 60)),  # Right.
            ((35, 15, 45, 25, .9), (59, 40)),  # Top.
            ((35, 55, 45, 65, .9), (59, 80)),  # Bottom.
        ):
            with self.subTest(box=box):
                self.assertEqual(self.locator([box]).locate(self.frame), target)

    def test_centers_just_outside_each_boundary_are_rejected(self):
        for box in (
            (14.99, 35, 24.99, 45, .9),
            (55.01, 35, 65.01, 45, .9),
            (35, 14.99, 45, 24.99, .9),
            (35, 55.01, 45, 65.01, .9),
        ):
            with self.subTest(box=box):
                self.assertIsNone(self.locator([box]).locate(self.frame))

    def test_valid_objects_keep_smallest_mechanical_center_x_order(self):
        # Reversed camera mapping: the lower-confidence, right-hand valid part
        # has the smallest mechanical X. The further-right part is outside.
        boxes = [(25, 35, 35, 45, .99), (45, 35, 55, 45, .8),
                 (65, 35, 75, 45, .95)]
        locator = self.locator(boxes, calibration=self.calibration(reverse_x=True))
        self.assertEqual(locator.locate(self.frame), (69, 60))

    def test_confidence_threshold_is_inclusive_for_an_in_frame_center(self):
        locator = self.locator([(35, 35, 45, 45, .5)], confidence=.5)
        self.assertEqual(locator.locate(self.frame), (59, 60))

    def test_invalid_region_is_rejected_at_construction(self):
        for region in (
            (), (0, 0, 1), (0, 0, 1, 1, 2), ((0, 0), (1, 1)), 4, '0123',
            (0, 0, 0, 1), (0, 0, 1, 0), (2, 0, 1, 1), (0, 2, 1, 1),
            (float('nan'), 0, 1, 1), (0, 0, float('inf'), 1), ('bad', 0, 1, 1),
        ):
            with self.subTest(region=region):
                with self.assertRaises(ValueError):
                    self.locator(selection_region=region)

    def test_invalid_decoded_boxes_are_rejected_even_with_in_frame_center(self):
        for box in (
            (-10, 35, 70, 45, .9), (15, 35, 95, 45, .9),
            (45, 35, 35, 45, .9), (35, 35, 35, 45, .9),
            (35, 35, 45, 45, float('nan')), (float('inf'), 35, 45, 45, .9),
        ):
            with self.subTest(box=box):
                self.assertIsNone(self.locator([box]).locate(self.frame))

    def test_outside_frame_candidate_is_skipped_before_physical_mapping(self):
        calibration = self.calibration(physical_region=(30, 40, 70, 80))
        # Strict mode would raise for this center at X=80 if it were mapped.
        locator = self.locator([(65, 35, 75, 45, .9)], calibration=calibration)
        self.assertIsNone(locator.locate(self.frame))

    def test_inside_frame_still_requires_a_safe_physical_center(self):
        locator = self.locator([(35, 35, 45, 45, .9)],
                               calibration=self.calibration(physical_region=(55, 40, 70, 80)),
                               skip_unsafe=True)
        self.assertIsNone(locator.locate(self.frame))

    def test_inside_frame_still_requires_a_safe_final_target(self):
        locator = self.locator([(55, 35, 65, 45, .9)],
                               calibration=self.calibration(physical_region=(30, 40, 78, 80)),
                               skip_unsafe=True)
        self.assertIsNone(locator.locate(self.frame))


if __name__ == '__main__':
    unittest.main()
