import unittest
from pathlib import Path
import cv2
import numpy as np
from tray_foreground import TrayForegroundDetector

FIXTURES=Path(__file__).parent/'fixtures/tray_glare_20261116'

class LiveFragmentTests(unittest.TestCase):
    def test_eight_live_frames_keep_five_parts_without_duplicate_dark_faces(self):
        paths=sorted(FIXTURES.glob('yellow_sequence_*.png'))
        self.assertEqual(len(paths),8)
        for path in paths:
            with self.subTest(frame=path.name):
                detector=TrayForegroundDetector(cv2.imread(str(FIXTURES/'reference.png')))
                boxes=detector.detect(cv2.imread(str(path)))
                self.assertEqual(len(boxes),5)
                self.assertEqual(detector.last_diagnostics['blocking_regions'],[])
                gray=[c for c in detector.last_diagnostics['candidates'] if c['group']=='cyan_gray']
                self.assertEqual(len(gray),1)
                self.assertTrue(250<=gray[0]['box_roi'][2]<=270)

    def setUp(self):
        self.detector=TrayForegroundDetector(cv2.imread(str(FIXTURES/'reference.png')))
        self.detector.detect(cv2.imread(str(FIXTURES/'yellow_loaded.png')))

    def test_yellow_internal_color_sliver_does_not_block_whole_cylinder(self):
        warm=[c for c in self.detector.last_diagnostics['candidates'] if c['group']=='warm']
        self.assertEqual(len(warm),1)
        x1,y1,x2,y2=warm[0]['box_roi']
        self.assertLessEqual(x1,163);self.assertGreaterEqual(x2,292)
        self.assertLessEqual(y1,120);self.assertGreaterEqual(y2,235)
        self.assertFalse([r for r in self.detector.last_diagnostics['blocking_regions'] if r['group']=='warm'])

    def test_soft_shadow_beside_yellow_is_not_an_obstacle(self):
        self.assertFalse([r for r in self.detector.last_diagnostics['blocking_regions'] if r['group']=='neutral_dark'])

    def test_gray_low_saturation_face_keeps_full_rear(self):
        gray=[c['box_roi'] for c in self.detector.last_diagnostics['candidates']
              if c['group']=='cyan_gray' and c['box_roi'][0]<200]
        self.assertEqual(len(gray),1)
        self.assertGreaterEqual(gray[0][2],250)
        self.assertLessEqual(gray[0][2],270)

    def test_lit_black_cylinder_is_one_body_without_joining_right_rail(self):
        dark=[c['box_roi'] for c in self.detector.last_diagnostics['candidates'] if c['group']=='dark_part']
        self.assertEqual(len(dark),2)
        right=max(dark,key=lambda b:b[0])
        self.assertLessEqual(right[0],454);self.assertGreaterEqual(right[2],610)
        self.assertGreaterEqual(right[1],150);self.assertLessEqual(right[3],315)
        self.assertFalse([r for r in self.detector.last_diagnostics['blocking_regions'] if r['group']=='dark_part'])

    def test_same_color_fragment_outside_actual_triangle_is_not_swallowed(self):
        ref=np.full((480,640,3),200,np.uint8);frame=ref.copy()
        cv2.fillPoly(frame,[np.array(((150,150),(150,310),(310,310)))],(30,160,30))
        cv2.rectangle(frame,(240,165),(250,200),(30,160,30),-1)
        detector=TrayForegroundDetector(ref);detector.detect(frame)
        self.assertTrue(detector.last_diagnostics['blocking_regions'])
