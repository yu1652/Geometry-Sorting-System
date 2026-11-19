"""Real tray snapshots: foreground pieces, dark rear faces and held-out empties."""
import json
from pathlib import Path
import unittest

import cv2
import numpy as np

try:
    from tray_foreground import TrayForegroundDetector
except ImportError:
    TrayForegroundDetector = None


FIXTURES = Path(__file__).parent / 'fixtures/tray_foreground'


def image(name):
    return cv2.imread(str(FIXTURES / (name + '.png')))


def overlap(a, b):
    ix = max(0, min(a[2],b[2])-max(a[0],b[0]))
    iy = max(0, min(a[3],b[3])-max(a[1],b[1]))
    intersection = ix*iy
    return intersection / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1])-intersection)


class TrayForegroundTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(TrayForegroundDetector, 'Foreground detector module is not implemented')
        self.reference = image('empty98')
        self.detector = TrayForegroundDetector(self.reference)

    def best(self, boxes, expected):
        self.assertTrue(boxes, 'Expected an independently observed part')
        index = max(range(len(boxes)), key=lambda i: overlap(boxes[i][:4],expected))
        return index, boxes[index], overlap(boxes[index][:4],expected)

    def test_first_touching_gray_and_green_are_distinct_and_keep_gray_rear_face(self):
        boxes = self.detector.detect(image('first36'))
        self.assertEqual(len(boxes),4,'Cable-occluded top rod must be blocking, not a partial push target')
        gray_i,gray,gray_iou = self.best(boxes,(77,244,225,373))
        green_i,green,green_iou = self.best(boxes,(65,362,220,479))
        self.assertNotEqual(gray_i,green_i,'Touching different pieces must not become a combined push')
        self.assertGreater(gray_iou,.80)
        self.assertGreater(green_iou,.75)
        self.assertLessEqual(abs(gray[2]-225),8,'Missing dark rear can land the pusher on the part')
        self.assertLessEqual(abs(green[2]-220),8)

    def test_full_frame_recovers_white_top_face_beyond_old_roi_right_edge(self):
        boxes = self.detector.detect(image('first36'))
        _,white,value = self.best(boxes,(465,28,564,126))
        self.assertGreater(value,.35)
        self.assertGreater(white[2],535)
        self.assertLessEqual(abs(white[2]-564),8)

    def test_gray81_remains_one_piece_with_dark_side_rear_included(self):
        for name in ('gray81','gray82'):
            with self.subTest(frame=name):
                boxes = self.detector.detect(image(name))
                self.assertEqual(len(boxes),1)
                self.assertGreater(overlap(boxes[0][:4],(166,245,315,389)),.80)
                self.assertLessEqual(abs(boxes[0][2]-315),8)
                self.assertEqual(self.detector.last_diagnostics['blocking_regions'],[])

    def test_rotated_gray51_and52_do_not_split_dark_side_into_second_part(self):
        for name in ('gray51','gray52'):
            with self.subTest(frame=name):
                boxes = self.detector.detect(image(name))
                self.assertEqual(len(boxes),3,'One yellow, one white cylinder and one gray cube')
                gray = [b for b in boxes if (b[1]+b[3])/2>250]
                self.assertEqual(len(gray),1)
                self.assertGreater(overlap(gray[0][:4],(169,249,336,415)),.80)
                self.assertLessEqual(abs(gray[0][2]-336),8)

    def test_empty_reference98_detects_independent_empty99_as_no_piece(self):
        self.assertEqual(self.detector.detect(image('empty99')),[])

    def test_empty_reference99_detects_independent_empty98_as_no_piece(self):
        detector = TrayForegroundDetector(image('empty99'))
        self.assertEqual(detector.detect(image('empty98')),[])

    def test_live_empty_highlight_appearance_and_disappearance_are_not_parts_or_blocks(self):
        for reference,current in (('live_reference','live_empty'),('live_empty','live_reference')):
            with self.subTest(reference=reference,current=current):
                detector=TrayForegroundDetector(image(reference))
                self.assertEqual(detector.detect(image(current)),[])
                self.assertEqual(detector.last_diagnostics['blocking_regions'],[])

    def test_strong_neutral_long_block_keeps_its_actual_boundary(self):
        ref=np.full((480,640,3),200,np.uint8)
        for level in (30,255):
            with self.subTest(level=level):
                frame=ref.copy()
                cv2.rectangle(frame,(150,150),(189,309),(level,level,level),-1)
                detector=TrayForegroundDetector(ref)
                boxes=detector.detect(frame)
                self.assertEqual(len(boxes),1)
                self.assertGreater(overlap(boxes[0][:4],(150,150,190,310)),.85)
                self.assertEqual(detector.last_diagnostics['blocking_regions'],[])

    def test_live_two_dark_plastic_parts_keep_full_bounds_and_left_column_priority(self):
        from tray_locator import TrayLocator
        from xy_calibration import CalibrationPoint,XYCalibration
        detector=TrayForegroundDetector(image('live_reference'))
        frame=image('live_two_dark')
        boxes=detector.detect(frame)
        self.assertEqual(len(boxes),2,'Low-brightness hue fragments belong to one actual connected part')
        left_i,left,left_iou=self.best(boxes,(103,206,251,329))
        upper_i,upper,upper_iou=self.best(boxes,(309,25,434,170))
        self.assertNotEqual(left_i,upper_i)
        self.assertGreater(left_iou,.90)
        self.assertGreater(upper_iou,.90)
        for actual,expected in ((left,(103,206,251,329)),(upper,(309,25,434,170))):
            self.assertTrue(all(abs(a-b)<=5 for a,b in zip(actual[:4],expected)),
                            'Full extent is needed for correct center and rear placement')
        self.assertEqual(detector.last_diagnostics['blocking_regions'],[])
        data=json.loads((FIXTURES.parent/'tray_20261116_push_trial.json').read_text(encoding='utf8'))
        calibration=XYCalibration.fit([CalibrationPoint(*p) for p in data['calibration_points']],
                                      pixel_region=(0,0,640,480),physical_region=(0,0,260,230))
        locator=TrayLocator(calibration,detector,(0,0,640,480),offsets=(1,0),confidence=.75,
                            column_boundaries=(95,152),motion_target_only=True,
                            compensation={'center':(80,80),'span':(80,80),'x_left':0,
                                          'x_right':-4,'y_top':6,'y_bottom':-6})
        target=locator.locate(frame)
        self.assertIsNotNone(target)
        trace=locator.last_diagnostics
        selected=trace['candidates'][trace['selected_index']]
        self.assertLess(selected['box_full'][0],300)
        self.assertEqual(selected['selection_key'][0],0)
        upper_record=next(c for c in trace['candidates'] if c['box_full'][0]>300)
        self.assertEqual(upper_record['selection_key'][0],1)

    def test_uniform_exposure_changes_keep_empty_and_gray_decisions(self):
        for factor in (.85,1.15):
            with self.subTest(factor=factor):
                empty=np.rint(np.clip(image('empty99').astype(float)*factor,0,255)).astype(np.uint8)
                gray=np.rint(np.clip(image('gray81').astype(float)*factor,0,255)).astype(np.uint8)
                self.assertEqual(self.detector.detect(empty),[])
                boxes=self.detector.detect(gray)
                self.assertEqual(len(boxes),1)
                self.assertLessEqual(abs(boxes[0][2]-315),8)

    def test_detector_accepts_matching_cropped_roi_and_returns_roi_coordinates(self):
        detector = TrayForegroundDetector(image('empty98')[:479,16:535])
        boxes = detector.detect(image('gray81')[:479,16:535])
        self.assertEqual(len(boxes),1)
        self.assertLessEqual(abs(boxes[0][2]-(315-16)),8)
        self.assertTrue(0<=boxes[0][0]<boxes[0][2]<=519)

    def test_reference_and_frame_are_preserved_and_diagnostics_are_json_except_mask(self):
        supplied = self.reference.copy()
        detector = TrayForegroundDetector(supplied)
        supplied[:] = 0
        frame=image('first36');before=frame.copy()
        boxes=detector.detect(frame)
        self.assertEqual(len(boxes),4)
        np.testing.assert_array_equal(frame,before)
        diagnostics=dict(detector.last_diagnostics)
        mask=diagnostics.pop('foreground_mask')
        self.assertEqual(mask.shape,(480,640))
        self.assertEqual(mask.dtype,np.uint8)
        self.assertEqual(len(diagnostics['candidates']),len(boxes))
        json.dumps(diagnostics,allow_nan=False)
        self.assertTrue(all(len(b)==5 and 0<=b[4]<=1 for b in boxes))

    def test_invalid_reference_and_mismatched_frame_are_rejected(self):
        for invalid in (None,np.zeros((20,20,3),np.uint8),np.zeros((480,640),np.uint8),
                        np.zeros((480,640,3),np.float32),np.zeros((480,640,3),np.uint8)):
            with self.subTest(reference_type=type(invalid)):
                with self.assertRaises(ValueError):
                    TrayForegroundDetector(invalid)
        for invalid in (None,np.zeros((479,640,3),np.uint8),np.zeros((480,640,3),np.float32)):
            with self.subTest(frame_type=type(invalid)):
                with self.assertRaises(ValueError):
                    self.detector.detect(invalid)

    def test_separate_purple_inside_triangle_bbox_is_not_swallowed(self):
        ref=np.full((480,640,3),200,np.uint8);frame=ref.copy()
        cv2.fillPoly(frame,[np.array(((150,150),(150,310),(310,310)))],(30,160,30))
        cv2.rectangle(frame,(235,165),(275,205),(180,80,180),-1)
        boxes=TrayForegroundDetector(ref).detect(frame)
        self.assertEqual(len(boxes),2,'Bounding-box containment is not physical contact')
        _,purple,value=self.best(boxes,(235,165,276,206))
        self.assertGreater(value,.70)

    def test_four_pixel_gap_separates_gray_and_dark_green_and_preserves_gray_rear(self):
        ref=np.full((480,640,3),200,np.uint8);frame=ref.copy()
        cv2.rectangle(frame,(150,150),(249,249),(180,180,100),-1)
        cv2.rectangle(frame,(254,155),(293,244),(90,120,90),-1)
        boxes=TrayForegroundDetector(ref).detect(frame)
        self.assertEqual(len(boxes),2,'Rear-face attachment cannot cross a four-pixel clear gap')
        _,gray,value=self.best(boxes,(150,150,250,250))
        self.assertGreater(value,.85)
        self.assertLessEqual(gray[2],252)

    def test_four_pixel_gap_separates_same_color_blocks(self):
        ref=np.full((480,640,3),200,np.uint8);frame=ref.copy()
        cv2.rectangle(frame,(150,150),(249,249),(30,160,30),-1)
        cv2.rectangle(frame,(254,150),(353,249),(30,160,30),-1)
        boxes=TrayForegroundDetector(ref).detect(frame)
        self.assertEqual(len(boxes),2,'Morphological closing must preserve the clear four-pixel gap')

    def test_four_pixel_gap_keeps_dark_parts_separate_despite_unstable_hue(self):
        for second in ((25,29,16),(29,25,16)):
            with self.subTest(second_color=second):
                ref=np.full((480,640,3),200,np.uint8);frame=ref.copy()
                cv2.rectangle(frame,(150,150),(249,249),(25,29,16),-1)
                cv2.rectangle(frame,(254,150),(353,249),second,-1)
                detector=TrayForegroundDetector(ref)
                boxes=detector.detect(frame)
                self.assertEqual(len(boxes),2)
                self.assertTrue(all(b[2]<=252 or b[0]>=252 for b in boxes))
                self.assertEqual(detector.last_diagnostics['blocking_regions'],[])

    def test_neutral_black_and_white_blocks_do_not_require_chroma_change(self):
        for level in (30,255):
            with self.subTest(level=level):
                ref=np.full((480,640,3),200,np.uint8);frame=ref.copy()
                cv2.rectangle(frame,(150,150),(249,249),(level,level,level),-1)
                boxes=TrayForegroundDetector(ref).detect(frame)
                self.assertEqual(len(boxes),1)
                self.assertGreater(overlap(boxes[0][:4],(150,150,250,250)),.85)

    def test_real_soft_glare61_is_illumination_while_overhanging_gray_stays_blocking(self):
        boxes=self.detector.detect(image('glare61'))
        self.assertEqual(len(boxes),1,'Overhanging gray is blocking and the glare is not a pushable piece')
        self.assertTrue(any(r.get('group')=='cyan_gray' for r in self.detector.last_diagnostics['blocking_regions']))
        self.assertTrue(any(r['reason']=='illumination_only_change'
                            and r['box_roi'][0]>480 for r in self.detector.last_diagnostics['rejected']))
        self.assertFalse(any(r.get('group')=='neutral_bright'
                             for r in self.detector.last_diagnostics['blocking_regions']))

    def test_oversized_joined_region_is_reported_as_blocking(self):
        ref=np.full((480,640,3),200,np.uint8);frame=ref.copy()
        cv2.rectangle(frame,(120,120),(420,260),(30,160,30),-1)
        detector=TrayForegroundDetector(ref)
        self.assertEqual(detector.detect(frame),[])
        self.assertIn('blocking_regions',detector.last_diagnostics)
        regions=detector.last_diagnostics['blocking_regions']
        self.assertTrue(any(r['reason']=='oversized_or_possible_joined_parts' for r in regions))

    def test_unobserved_rear_at_right_image_edge_is_reported_as_blocking(self):
        ref=np.full((480,640,3),200,np.uint8);frame=ref.copy()
        cv2.rectangle(frame,(600,200),(639,300),(30,160,30),-1)
        detector=TrayForegroundDetector(ref)
        self.assertEqual(detector.detect(frame),[])
        self.assertTrue(any(r['reason']=='rear_clipped' for r in detector.last_diagnostics['blocking_regions']))

    def test_significant_clipped_sliver_and_long_joined_region_are_not_silently_filtered(self):
        for rectangle in ((630,200,639,300),(20,200,620,250)):
            with self.subTest(rectangle=rectangle):
                ref=np.full((480,640,3),200,np.uint8);frame=ref.copy()
                cv2.rectangle(frame,rectangle[:2],rectangle[2:],(30,160,30),-1)
                detector=TrayForegroundDetector(ref)
                self.assertEqual(detector.detect(frame),[])
                self.assertTrue(detector.last_diagnostics['blocking_regions'])

    def test_real_cable_split_rod_is_blocking_instead_of_using_truncated_center(self):
        boxes=self.detector.detect(image('first36'))
        self.assertFalse(any(b[0]<230 and b[1]<155 and b[3]<180 for b in boxes))
        regions=[r for r in self.detector.last_diagnostics['blocking_regions']
                 if r['reason'] in ('reference_mask_truncates_part','occluded_or_split_part')]
        self.assertTrue(regions)
        self.assertTrue(any(r['box_roi'][0]<=126 and r['box_roi'][1]<=28
                            and r['box_roi'][2]>=215 and r['box_roi'][3]>=148 for r in regions))

    def test_reference_surface_concavity_cannot_authorize_a_truncated_rear(self):
        ref=np.full((480,640,3),80,np.uint8)
        cv2.rectangle(ref,(100,100),(500,400),(200,200,200),-1)
        frame=ref.copy();cv2.rectangle(frame,(480,200),(550,300),(30,160,160),-1)
        detector=TrayForegroundDetector(ref)
        self.assertEqual(detector.detect(frame),[])
        self.assertTrue(any(r['reason']=='reference_mask_truncates_part'
                            for r in detector.last_diagnostics['blocking_regions']))

    def test_reference_surface_vertical_cut_cannot_shift_a_part_center_into_motion_range(self):
        for direction in ('top','bottom','left','right'):
            with self.subTest(direction=direction):
                ref=np.full((480,640,3),200,np.uint8)
                if direction=='top':ref[:200,100:300]=80
                elif direction=='bottom':ref[200:,100:300]=80
                elif direction=='left':ref[100:300,:200]=80
                else:ref[100:300,200:]=80
                frame=ref.copy();cv2.rectangle(frame,(150,150),(249,249),(30,160,160),-1)
                detector=TrayForegroundDetector(ref)
                boxes=detector.detect(frame)
                if boxes:
                    self.assertEqual(len(boxes),1)
                    self.assertGreater(overlap(boxes[0][:4],(150,150,250,250)),.85)
                    self.assertLessEqual(abs((boxes[0][1]+boxes[0][3])/2-200),3)
                    self.assertLessEqual(abs((boxes[0][0]+boxes[0][2])/2-200),3)
                else:
                    self.assertTrue(detector.last_diagnostics['blocking_regions'])

    def test_unstable_match_has_hold_reason_and_small_reference_surface_is_rejected(self):
        ref=np.full((480,640,3),200,np.uint8)
        detector=TrayForegroundDetector(ref)
        self.assertEqual(detector.detect(np.zeros_like(ref)),[])
        self.assertIn('hold_reason',detector.last_diagnostics)
        self.assertEqual(detector.last_diagnostics['hold_reason'],'底板参考无法稳定匹配，请保持取景或重新保存底板')
        small=np.zeros_like(ref);cv2.rectangle(small,(200,200),(230,230),(200,200,200),-1)
        with self.assertRaises(ValueError):
            TrayForegroundDetector(small)


if __name__=='__main__':
    unittest.main()
