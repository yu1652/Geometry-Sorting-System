import unittest
import numpy as np
from xy_calibration import CalibrationPoint, XYCalibration
from tray_locator import TrayLocator

class GeometryTests(unittest.TestCase):
    def calibration(self):
        points = [CalibrationPoint(x,y,x,y) for x in (0,5,10) for y in (0,5,10)]
        return XYCalibration.fit(points, pixel_region=(0,0,10,10), physical_region=(0,0,20,20))
    def test_mapping_and_safety(self):
        c = self.calibration()
        np.testing.assert_allclose(c.map_pixel(3,4), (3,4), atol=1e-9)
        with self.assertRaises(ValueError): c.map_pixel(11,4)
        with self.assertRaises(ValueError): XYCalibration.fit([CalibrationPoint(0,0,0,0)]*9, pixel_region=(0,0,10,10), physical_region=(0,0,20,20))
    def test_selection_and_offset(self):
        class Detector:
            def detect(self, image): return [(5,1,7,3,.9), (1,2,3,4,.8)]
        result = TrayLocator(self.calibration(), Detector(), (0,0,10,10)).locate(np.zeros((10,10,3)))
        np.testing.assert_allclose(result, (7,3), atol=1e-9)

if __name__ == '__main__': unittest.main()

class SafetyTests(GeometryTests):
    def test_rejects_nonfinite_rank_and_physical_offset(self):
        with self.assertRaises(ValueError):
            XYCalibration.fit([CalibrationPoint(x, 0, x, 0) for x in range(6)], pixel_region=(0,0,10,10), physical_region=(0,0,20,20))
        with self.assertRaises(ValueError): self.calibration().map_pixel(float('nan'), 1)
        class Detector:
            def detect(self, image): return [(8,1,10,3,.9)]
        with self.assertRaises(ValueError):
            TrayLocator(self.calibration(), Detector(), (0,0,10,10), offsets=(20,0)).locate(np.zeros((10,10,3)))

class TrayDetectorTests(unittest.TestCase):
    def test_decode_and_nms(self):
        from tray_locator import TrayDetector
        class Backend:
            def infer(self, tensor):
                assert tensor.shape == (1,3,640,640)
                output = np.zeros((1,5,8400), dtype=np.float32)
                output[0,:,0] = (320,320,200,100,.9)
                output[0,:,1] = (320,320,200,100,.8)
                return output
        boxes = TrayDetector(Backend()).detect(np.zeros((480,640,3), dtype=np.uint8))
        self.assertEqual(len(boxes), 1)
        np.testing.assert_allclose(boxes[0][:4], (220,190,420,290))

    def test_feature_input_changes_model_tensor_before_letterboxing(self):
        from tray_locator import TrayDetector

        class Backend:
            def infer(self, tensor):
                self.tensor = tensor
                return np.zeros((1, 5, 8400), dtype=np.float32)

        frame = np.zeros((87, 152, 3), dtype=np.uint8)
        frame[20:60, 30:100] = (51, 113, 219)
        raw, feature = Backend(), Backend()
        TrayDetector(raw).detect(frame)
        TrayDetector(feature, feature_input=True).detect(frame)

        from feature_preprocess import feature_bgr
        from vision import letterbox_rgb
        np.testing.assert_array_equal(raw.tensor, letterbox_rgb(frame, size=640)[0])
        np.testing.assert_array_equal(feature.tensor, letterbox_rgb(feature_bgr(frame), size=640)[0])
        self.assertFalse(np.array_equal(raw.tensor, feature.tensor))


class TrayCompensationTests(unittest.TestCase):
    compensation = {
        'center': [80, 80], 'span': [80, 80],
        'x_left': 0, 'x_right': -4, 'y_top': 6, 'y_bottom': -6,
    }

    def setUp(self):
        points = [CalibrationPoint(x, y, x, y) for x in (10, 110, 210) for y in (10, 110, 210)]
        self.calibration = XYCalibration.fit(points, pixel_region=(0, 0, 220, 220),
                                             physical_region=(0, 0, 220, 220))
        self.frame = np.zeros((220, 220, 3), dtype=np.uint8)

    def locate(self, x, y, offsets=(4, 0)):
        class Detector:
            def detect(self, image):
                # A more confident detection to the right must still lose to the leftmost piece.
                return [(x - 1, y - 1, x + 2, y + 1, .99), (x - 5, y - 1, x, y + 1, .8)]

        return TrayLocator(self.calibration, Detector(), (0, 0, 220, 220), offsets=offsets,
                           compensation=self.compensation).locate(self.frame)

    def test_position_compensation_is_centered_linear_and_bounded(self):
        for position, expected in (
            ((80, 80), (84, 80)),
            ((40, 40), (44, 43)),
            ((120, 120), (124, 117)),
            ((160, 160), (164, 154)),
            ((200, 200), (204, 194)),
        ):
            with self.subTest(position=position):
                np.testing.assert_allclose(self.locate(*position), expected, atol=1e-9)

    def test_position_compensation_validates_combined_target(self):
        # Each correction alone is legal, but the combined Y would be 221.5 mm.
        with self.assertRaisesRegex(ValueError, 'outside physical region'):
            self.locate(120, 40, offsets=(4, 178.5))

    def test_optional_safe_selection_skips_invalid_center_edge_and_corrected_target(self):
        points = [CalibrationPoint(x, y, x, y) for x in (20, 110, 200) for y in (20, 110, 200)]
        calibration = XYCalibration.fit(points, pixel_region=(0, 0, 220, 220),
                                        physical_region=(10, 10, 210, 210))

        class Detector:
            def detect(self, image):
                return self.detections

        detector = Detector()
        settings = dict(offsets=(4, 13), compensation=self.compensation)
        safe = TrayLocator(calibration, detector, (0, 0, 220, 220), skip_unsafe=True, **settings)
        strict = TrayLocator(calibration, detector, (0, 0, 220, 220), **settings)
        for unsafe in (
            (0, 40, 8, 60, .8),       # Center X is outside the physical region.
            (20, 30, 220, 50, .8),   # Edge X is outside the physical region.
            (20, 200, 40, 208, .8),  # Edge Y is valid until the offsets raise it to 211.
        ):
            with self.subTest(unsafe=unsafe):
                detector.detections = [(135, 75, 160, 85, .9), unsafe]
                with self.assertRaisesRegex(ValueError, 'outside physical region'):
                    strict.locate(self.frame)
                np.testing.assert_allclose(safe.locate(self.frame), (164, 93), atol=1e-9)
                detector.detections = [unsafe]
                self.assertIsNone(safe.locate(self.frame))


class PushRearEdgeTests(unittest.TestCase):
    def calibration(self, mapping, physical_region=(0, 0, 300, 300)):
        return XYCalibration.fit(
            [CalibrationPoint(x, y, *mapping(x, y)) for x in (0, 100, 200) for y in (0, 100, 200)],
            pixel_region=(0, 0, 220, 220), physical_region=physical_region)

    def locate(self, calibration, boxes, *, offsets=(4, 0), compensation=None, skip_unsafe=False):
        class Detector:
            def detect(self, image):
                return boxes
        return TrayLocator(calibration, Detector(), (0, 0, 220, 220), offsets,
                           compensation=compensation, skip_unsafe=skip_unsafe).locate(
                               np.zeros((220, 220, 3), np.uint8))

    def test_forward_mapping_keeps_right_rear_edge_and_center_y(self):
        calibration = self.calibration(lambda x, y: (x, y))
        np.testing.assert_allclose(self.locate(calibration, [(40, 70, 80, 110, .9)]),
                                   (84, 90), atol=1e-8)

    def test_reversed_mapping_uses_left_image_edge_as_physical_rear(self):
        calibration = self.calibration(lambda x, y: (220 - x, 220 - y))
        np.testing.assert_allclose(self.locate(calibration, [(40, 70, 80, 110, .9)]),
                                   (184, 130), atol=1e-8)

    def test_quarter_turn_uses_four_corners_instead_of_left_right_midpoints(self):
        calibration = self.calibration(lambda x, y: (y, 220 - x))
        np.testing.assert_allclose(self.locate(calibration, [(40, 70, 80, 110, .9)]),
                                   (114, 160), atol=1e-8)

    def test_oblique_box_uses_maximum_x_corner_but_y_at_box_center(self):
        calibration = self.calibration(lambda x, y: (.8 * x + .4 * y, 60 - .4 * x + .8 * y))
        np.testing.assert_allclose(self.locate(calibration, [(40, 70, 80, 110, .9)]),
                                   (112, 108), atol=1e-8)

    def test_negative_old_compensation_cannot_cross_rear_gap_and_positive_still_applies(self):
        calibration = self.calibration(lambda x, y: (x, y))
        settings = dict(TrayCompensationTests.compensation)
        box = [(100, 100, 120, 120, .9)]
        np.testing.assert_allclose(self.locate(calibration, box, compensation=settings),
                                   (124, 107.75), atol=1e-8)
        settings['x_right'] = 4
        np.testing.assert_allclose(self.locate(calibration, box, compensation=settings),
                                   (126, 107.75), atol=1e-8)

    def test_unsafe_rear_target_skips_candidate_without_clamping_in_front(self):
        calibration = self.calibration(lambda x, y: (220 - x, y), physical_region=(0, 0, 200, 220))
        # Wide first candidate has the smaller physical center X, but its rear
        # plus the configured gap is outside the range. Select the safe piece.
        boxes = [(40, 50, 200, 70, .9), (60, 80, 80, 100, .8)]
        np.testing.assert_allclose(self.locate(calibration, boxes, offsets=(30, 0), skip_unsafe=True),
                                   (190, 90), atol=1e-8)
        self.assertIsNone(self.locate(calibration, boxes[:1], offsets=(30, 0), skip_unsafe=True))
        with self.assertRaisesRegex(ValueError, 'outside physical region'):
            self.locate(calibration, boxes[:1], offsets=(30, 0))

    def test_multiple_candidates_keep_negative_x_physical_push_order(self):
        calibration = self.calibration(lambda x, y: (220 - x, y))
        np.testing.assert_allclose(self.locate(calibration,
                                   [(20, 50, 40, 70, .99), (140, 80, 160, 100, .8)]),
                                   (84, 90), atol=1e-8)

    def test_latest_reversed_field_calibration_does_not_pick_truncated_front_edge(self):
        # Nine manually adjusted points from the installed board; RGB silhouette
        # has its green part around x375..550 and extends past the 535px ROI edge.
        rows = [
            (217.009, 110.4975, 181, 125), (351.03, 109.4975, 124.2, 124),
            (486.051, 109.4975, 66, 124), (216.009, 243.325, 180, 66),
            (350.03, 245.325, 123, 66), (486.051, 243.325, 66, 66),
            (216.009, 377.1525, 180, 9), (350.03, 377.1525, 123, 9),
            (483.051, 377.1525, 67, 9),
        ]
        calibration = XYCalibration.fit([CalibrationPoint(*row) for row in rows],
            pixel_region=(16, 0, 535, 479), physical_region=(43.563, .125, 226.331, 164.027))

        class Detector:
            def detect(self, image):
                return [(359, 269, 519, 432.478, .9)]

        locator = TrayLocator(calibration, Detector(), (16, 0, 519, 479), offsets=(1, 0),
                               compensation=TrayCompensationTests.compensation, skip_unsafe=True)
        result = locator.locate(np.zeros((480, 640, 3), np.uint8))
        self.assertIsNotNone(result)
        # Legacy fixed-right calculation exactly reproduced PUSH 46.081,24.729;
        # this candidate must start behind the part near X113, not its front.
        self.assertGreater(result[0], 110)
        projected = [calibration.project_pixel(x, y)[0]
                     for x in (375, 535) for y in (269, 432.478)]
        self.assertGreaterEqual(result[0] + 1e-9, max(projected) + 1)
        self.assertAlmostEqual(result[1], 24.750955, delta=.03)
        self.assertLess(calibration.project_pixel(375, 432.478)[1], 0)

    def test_projected_corner_outside_motion_range_does_not_grant_motion_authority(self):
        calibration = self.calibration(lambda x, y: (x, y), physical_region=(0, 40, 200, 160))
        np.testing.assert_allclose(calibration.project_pixel(60, 180), (60, 180), atol=1e-8)
        with self.assertRaisesRegex(ValueError, 'outside physical region'):
            calibration.map_pixel(60, 180)
        with self.assertRaisesRegex(ValueError, 'outside safe region'):
            calibration.project_pixel(221, 100)
        with self.assertRaisesRegex(ValueError, 'outside physical region'):
            self.locate(calibration, [(40, 155, 80, 185, .9)])
