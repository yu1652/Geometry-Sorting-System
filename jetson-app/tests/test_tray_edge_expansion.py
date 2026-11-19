"""Complete observed edge parts without admitting detached outside clutter."""
import unittest
import cv2
import numpy as np
from tray_foreground import TrayForegroundDetector


class TrayEdgeExpansionTests(unittest.TestCase):
    def reference(self):
        ref = np.full((480, 640, 3), 80, np.uint8)
        cv2.rectangle(ref, (100, 100), (500, 400), (200, 200, 200), -1)
        return ref

    def test_observed_part_28_pixels_beyond_floor_keeps_complete_rear(self):
        ref = self.reference()
        frame = ref.copy()
        cv2.rectangle(frame, (450, 200), (528, 280), (30, 160, 30), -1)
        detector = TrayForegroundDetector(ref)
        boxes = detector.detect(frame)
        self.assertEqual(len(boxes), 1, detector.last_diagnostics)
        self.assertGreaterEqual(boxes[0][2], 529)
        self.assertEqual(detector.last_diagnostics['blocking_regions'], [])

    def test_detached_outside_part_cannot_seed_a_target(self):
        ref = self.reference()
        frame = ref.copy()
        cv2.rectangle(frame, (505, 200), (530, 280), (30, 160, 30), -1)
        self.assertEqual(TrayForegroundDetector(ref).detect(frame), [])
