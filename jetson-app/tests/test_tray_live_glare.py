import unittest
from pathlib import Path
import cv2
import numpy as np
from tray_foreground import TrayForegroundDetector

FIXTURES=Path(__file__).parent/'fixtures/tray_glare_20261116'

class LiveGlareTests(unittest.TestCase):
    def test_live_empty_rails_and_cable_shadow_never_block(self):
        detector=TrayForegroundDetector(cv2.imread(str(FIXTURES/'reference.png')))
        self.assertEqual(detector.detect(cv2.imread(str(FIXTURES/'empty.png'))),[])
        self.assertEqual(detector.last_diagnostics['blocking_regions'],[])

    def test_sharp_neutral_clipped_parts_still_block(self):
        ref=np.full((480,640,3),200,np.uint8)
        for level in (30,255):
            frame=ref.copy();cv2.rectangle(frame,(630,200),(639,350),(level,level,level),-1)
            detector=TrayForegroundDetector(ref)
            self.assertEqual(detector.detect(frame),[])
            self.assertTrue(detector.last_diagnostics['blocking_regions'])

    def test_actual_lamp_reflections_do_not_become_front_obstacles(self):
        ref=cv2.imread(str(FIXTURES/'reference.png'))
        for name in ('loaded','later'):
            with self.subTest(name=name):
                detector=TrayForegroundDetector(ref)
                boxes=detector.detect(cv2.imread(str(FIXTURES/(name+'.png'))))
                self.assertGreaterEqual(len(boxes),3)
                self.assertFalse([r for r in detector.last_diagnostics['blocking_regions']
                                  if r['group']=='neutral_bright'])
                groups={r['group'] for r in detector.last_diagnostics['candidates']}
                self.assertTrue({'green','dark_part','cyan_gray'} <= groups)
                if name=='loaded':
                    self.assertTrue(any(r['group']=='warm' for r in detector.last_diagnostics['blocking_regions']),
                                    'Touching real yellow parts must remain unresolved')

    def test_low_contrast_white_solid_and_strong_irregular_white_are_retained(self):
        ref=np.full((480,640,3),200,np.uint8)
        for level,notch in ((230,False),(255,True)):
            with self.subTest(level=level):
                frame=ref.copy()
                cv2.rectangle(frame,(150,150),(249,249),(level,level,level),-1)
                if notch: cv2.rectangle(frame,(200,150),(249,199),(200,200,200),-1)
                self.assertEqual(len(TrayForegroundDetector(ref).detect(frame)),1)
