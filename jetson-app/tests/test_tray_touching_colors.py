import unittest
from pathlib import Path
import cv2
import numpy as np
from tray_foreground import TrayForegroundDetector

FIXTURES=Path(__file__).parent/'fixtures/tray_glare_20261116'

class TouchingColorTests(unittest.TestCase):
    def test_model_hints_separate_three_observed_blue_parts_in_live_frame(self):
        fixtures = FIXTURES.parent / 'tray_overlap_20261118'
        reference = cv2.imread(str(fixtures / 'reference.png'))
        frame = cv2.imread(str(fixtures / 'loaded.png'))
        hints = [(310.62, 19.75, 416.38, 112.25, .86),
                 (156.0, 141.44, 276.0, 248.56, .85),
                 (223.38, 67.56, 378.62, 201.44, .74)]
        detector = TrayForegroundDetector(reference, pale_hints=hints)
        detector.detect(frame)
        parts = [c for c in detector.last_diagnostics['candidates']
                 if c['reason'] == 'model_and_visible_color_boundary']
        self.assertEqual(len(parts), 3)
        for expected in ((316, 26, 411, 112), (168, 142, 271, 249), (226, 67, 378, 200)):
            self.assertTrue(any(np.allclose(c['box_roi'], expected, atol=5) for c in parts), expected)
        self.assertFalse([r for r in detector.last_diagnostics['blocking_regions']
                          if r['group'] == 'cyan_color'])

    def test_two_model_boxes_cannot_split_a_uniform_joined_region(self):
        reference = np.full((480, 640, 3), 200, np.uint8)
        frame = reference.copy()
        cv2.rectangle(frame, (150, 140), (399, 259), (215, 150, 20), -1)
        detector = TrayForegroundDetector(reference, pale_hints=[(148, 138, 280, 262, .93),
                                                                 (269, 138, 402, 262, .92)])
        self.assertEqual(detector.detect(frame), [])
        self.assertTrue(any(r['reason'] == 'oversized_or_possible_joined_parts'
                            for r in detector.last_diagnostics['blocking_regions']))

    def test_two_strong_hints_can_seed_a_third_part_with_a_visible_seam(self):
        fixtures = FIXTURES.parent / 'tray_overlap_20261118'
        reference = cv2.imread(str(fixtures / 'reference.png'))
        frame = cv2.imread(str(fixtures / 'secondary_hint.png'))
        hints = [(315.5, 240.9, 427.5, 353.1, .856),
                 (459.0, 148.1, 590.0, 253.9, .764),
                 (331.1, 142.1, 462.9, 244.4, .610)]
        detector = TrayForegroundDetector(reference, pale_hints=hints)
        detector.detect(frame)
        parts = [c['box_roi'] for c in detector.last_diagnostics['candidates']
                 if c['reason'] == 'model_and_visible_color_boundary']
        self.assertEqual(len(parts), 3)
        for expected in ((315, 234, 433, 354), (454, 142, 598, 251), (333, 145, 470, 252)):
            self.assertTrue(any(np.allclose(box, expected, atol=5) for box in parts), expected)

        hints[1] = hints[1][:4] + (.70,)
        detector = TrayForegroundDetector(reference, pale_hints=hints)
        detector.detect(frame)
        self.assertFalse(any(c['reason'] == 'model_and_visible_color_boundary'
                             for c in detector.last_diagnostics['candidates']))

    def test_touching_white_gray_and_black_parts_split_at_visible_seams(self):
        reference = np.full((480, 640, 3), 200, np.uint8)
        hints = [(148, 128, 242, 226, .92), (236, 168, 280, 211, .90)]
        for bright, dark in ((245, 225), (115, 82), (24, 12)):
            with self.subTest(large=bright, small=dark):
                frame = reference.copy()
                frame[130:224, 150:240] = bright
                frame[170:209, 240:278] = dark
                detector = TrayForegroundDetector(reference, pale_hints=hints)
                boxes = detector.detect(frame)
                self.assertEqual(len(boxes), 2)
                self.assertTrue(all(c['reason'] == 'model_and_visible_color_boundary'
                                    for c in detector.last_diagnostics['candidates']))
                self.assertEqual(detector.last_diagnostics['blocking_regions'], [])

    def test_both_layouts_survive_exposure_changes_and_detector_restarts(self):
        ref=cv2.imread(str(FIXTURES/'reference.png'))
        for prefix,count in (('yellow_sequence_',5),('seven_sequence_',7)):
            for path in sorted(FIXTURES.glob(prefix+'*.png')):
                for factor in (.85,1.15):
                    with self.subTest(frame=path.name,exposure=factor):
                        frame=np.clip(cv2.imread(str(path)).astype(float)*factor,0,255).astype(np.uint8)
                        detector=TrayForegroundDetector(ref)
                        self.assertEqual(len(detector.detect(frame)),count)
                        self.assertEqual(detector.last_diagnostics['blocking_regions'],[])

    def test_sharp_pale_gray_piece_is_retained(self):
        ref=np.full((480,640,3),200,np.uint8)
        hsv=cv2.cvtColor(ref,cv2.COLOR_BGR2HSV)
        hsv[150:250,150:250]=(100,40,130)
        detector=TrayForegroundDetector(ref)
        self.assertEqual(len(detector.detect(cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR))),1)
        self.assertEqual(detector.last_diagnostics['blocking_regions'],[])

    def test_same_hue_gray_and_saturated_blue_remain_two_touching_parts(self):
        ref=np.full((480,640,3),200,np.uint8)
        hsv=cv2.cvtColor(ref,cv2.COLOR_BGR2HSV)
        hsv[150:250,150:250]=(100,75,130)
        hsv[150:250,250:350]=(100,190,180)
        detector=TrayForegroundDetector(ref)
        boxes=detector.detect(cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR))
        self.assertEqual(len(boxes),2)
        self.assertEqual(detector.last_diagnostics['blocking_regions'],[])
        self.assertTrue(all(b[2]<=253 or b[0]>=247 for b in boxes))

    def test_seven_real_parts_remain_separate_after_cable_removed_from_reference(self):
        paths=sorted(FIXTURES.glob('seven_sequence_*.png'))
        self.assertEqual(len(paths),8)
        for path in paths:
            with self.subTest(frame=path.name):
                detector=TrayForegroundDetector(cv2.imread(str(FIXTURES/'reference.png')))
                boxes=detector.detect(cv2.imread(str(path)))
                self.assertEqual(len(boxes),7)
                self.assertEqual(detector.last_diagnostics['blocking_regions'],[])
                # All detections must match one observed solid, not the rail,
                # the removed cable, or a union of adjacent solids.
                expected=[(94,76,275,227),(208,4,352,145),(322,26,484,170),
                          (454,158,606,304),(286,212,421,342),
                          (175,273,312,405),(382,377,503,465)]
                matched=set()
                for box in boxes:
                    scores=[]
                    for target in expected:
                        ix=max(0,min(box[2],target[2])-max(box[0],target[0]))
                        iy=max(0,min(box[3],target[3])-max(box[1],target[1]))
                        union=(box[2]-box[0])*(box[3]-box[1])+(target[2]-target[0])*(target[3]-target[1])-ix*iy
                        scores.append(ix*iy/union)
                    best=int(np.argmax(scores))
                    self.assertGreater(scores[best],.75)
                    self.assertNotIn(best,matched);matched.add(best)
