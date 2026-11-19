import unittest
from pathlib import Path

import cv2
import numpy as np

from tray_foreground import TrayForegroundDetector


FIXTURES = Path(__file__).parent / 'fixtures/tray_cable_20261117'


def cable_frame(reference, offset=0):
    frame = reference.copy()
    points = np.array(((0, 430), (70+offset, 340), (145+offset, 210),
                       (160+offset, 80), (170+offset, 0)), np.int32)
    cv2.polylines(frame, [points], False, (25, 25, 25), 20)
    cv2.polylines(frame, [points], False, (35, 35, 180), 10)
    return frame


class ServoCableTests(unittest.TestCase):
    def test_moving_edge_connected_wire_is_neither_target_nor_blocker(self):
        reference = np.full((480, 640, 3), 200, np.uint8)
        for offset in (0, 25, 70):
            with self.subTest(offset=offset):
                detector = TrayForegroundDetector(reference)
                self.assertEqual(detector.detect(cable_frame(reference, offset)), [])
                self.assertEqual(detector.last_diagnostics['blocking_regions'], [])
                self.assertTrue(detector.last_diagnostics.get('excluded_servo_cables'))

    def test_reference_cutting_a_wire_does_not_promote_it_to_incomplete_part(self):
        reference = np.full((480, 640, 3), 200, np.uint8)
        reference[:130, 110:200] = 30
        frame = cable_frame(np.full_like(reference, 200))
        detector = TrayForegroundDetector(reference)
        detector.detect(frame)
        self.assertFalse([r for r in detector.last_diagnostics['blocking_regions']
                          if r['group'] in ('warm', 'red_wrap', 'dark_part')])

    def test_real_red_parts_are_retained_next_to_wire_and_at_frame_edge(self):
        reference = np.full((480, 640, 3), 200, np.uint8)
        for x in (0, 300):
            with self.subTest(x=x):
                frame = cable_frame(reference)
                cv2.rectangle(frame, (x, 150), (x+65, 215), (35, 35, 180), -1)
                detector = TrayForegroundDetector(reference)
                boxes = detector.detect(frame)
                self.assertTrue(any(b[0] <= x+32 <= b[2] and b[1] <= 180 <= b[3]
                                    for b in boxes))

    def test_cable_touching_red_part_does_not_erase_the_part(self):
        reference = np.full((480, 640, 3), 200, np.uint8)
        frame = cable_frame(reference)
        cv2.rectangle(frame, (120, 175), (190, 245), (35, 35, 180), -1)
        detector = TrayForegroundDetector(reference)
        boxes = detector.detect(frame)
        regions = [list(b[:4]) for b in boxes]
        regions += [r['box_roi'] for r in detector.last_diagnostics['blocking_regions']]
        self.assertTrue(any(x1 <= 155 <= x2 and y1 <= 210 <= y2 for x1,y1,x2,y2 in regions))

    def test_small_independent_colored_part_beside_wire_is_not_a_strand(self):
        reference = np.full((480, 640, 3), 200, np.uint8)
        for color,height in [(c,h) for c in ((180,60,30),(35,35,180),(30,160,30)) for h in (40,65)]:
            with self.subTest(color=color,height=height):
                frame = reference.copy()
                cv2.line(frame, (100,0), (100,400), (35,35,180), 12)
                # Same-color objects need a gap wider than the pre-existing
                # 3px closing operator; otherwise they are already joined.
                left = 112 if color == (35,35,180) else 108
                cv2.rectangle(frame, (left,150), (left+15,150+height), color, -1)
                detector = TrayForegroundDetector(reference)
                boxes = detector.detect(frame)
                self.assertTrue(any(b[0] <= left+7 <= b[2] and b[1] <= 170 <= b[3]
                                    for b in boxes))

    def test_recorded_cable_is_not_a_push_target_and_solid_parts_survive(self):
        reference = cv2.imread(str(FIXTURES / 'reference.png'))
        for path in sorted(FIXTURES.glob('live_*.png')):
            with self.subTest(frame=path.name):
                detector = TrayForegroundDetector(reference)
                boxes = detector.detect(cv2.imread(str(path)))
                self.assertFalse([b for b in boxes if b[2] < 190 and b[1] < 250])
                self.assertTrue(detector.last_diagnostics.get('excluded_servo_cables'))
                # The yellow cylinder and blue part remain observed.
                for px,py in ((400,160), (300,390)):
                    self.assertTrue(any(b[0] <= px <= b[2] and b[1] <= py <= b[3] for b in boxes))

    def test_compact_dark_part_at_edge_next_to_lead_is_not_erased(self):
        reference = np.full((480,640,3),200,np.uint8)
        frame = reference.copy()
        cv2.line(frame,(100,0),(100,400),(35,35,180),12)
        cv2.rectangle(frame,(108,0),(128,80),(25,25,25),-1)
        detector=TrayForegroundDetector(reference)
        boxes=detector.detect(frame)
        self.assertTrue(any(b[0]<=118<=b[2] and b[1]<=40<=b[3] for b in boxes))


if __name__ == '__main__':
    unittest.main()
