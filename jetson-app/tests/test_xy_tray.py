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
