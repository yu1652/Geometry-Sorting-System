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
            ((120, 120), (122, 117)),
            ((160, 160), (160, 154)),
            ((200, 200), (200, 194)),
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
                np.testing.assert_allclose(safe.locate(self.frame), (160, 93), atol=1e-9)
                detector.detections = [unsafe]
                self.assertIsNone(safe.locate(self.frame))
