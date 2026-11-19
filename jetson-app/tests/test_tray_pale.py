"""Recorded white parts must survive weak contrast without promoting glare."""
import json
from pathlib import Path
import unittest
import cv2
import numpy as np
from tray_foreground import TrayForegroundDetector

FIX = Path(__file__).parent/'fixtures/tray_white_20261117'


class PalePartTests(unittest.TestCase):
    def detect(self, frame, hints, reference=None):
        if reference is None:
            reference = cv2.imread(str(FIX/'reference.png'))
        detector = TrayForegroundDetector(reference, pale_hints=hints)
        detector.detect(frame)
        return detector.last_diagnostics

    def test_real_white_parts_recovered_or_retained_as_ambiguous_obstacles(self):
        records = json.loads((FIX/'hints.json').read_text())
        centers = [(297,98),(200,200),(291,290),(435,425),(487,105)]
        for i, record in enumerate(records):
            with self.subTest(frame=i):
                trace = self.detect(cv2.imread(str(FIX/('live%d.png'%i))), record['full'])
                pale = [c for c in trace['candidates'] if c['group']=='model_pale']
                self.assertEqual(len(pale),4)
                observed=pale+[c for c in trace['blocking_regions'] if c['group']=='model_pale']
                for x,y in centers:
                    self.assertEqual(sum(b['box_roi'][0]<=x<=b['box_roi'][2] and
                                         b['box_roi'][1]<=y<=b['box_roi'][3] for b in observed),1)
                self.assertFalse(any(c['group']=='neutral_bright' for c in trace['candidates']))
                for c in pale:
                    box=c['box_roi']; hint=c['model_box_roi']
                    self.assertLessEqual(box[0],hint[0]); self.assertGreaterEqual(box[2],hint[2])
                    self.assertEqual(c['footprint_roi'],[[box[0],box[1]],[box[2],box[1]],
                                                        [box[2],box[3]],[box[0],box[3]]])

    def test_model_boxes_alone_cannot_promote_empty_tray_or_reflections(self):
        folder=FIX.parent/'tray_glare_20261116'
        reference=cv2.imread(str(folder/'reference.png'))
        hints=json.loads((FIX/'hints.json').read_text())[-1]['full']
        for name in ('reference.png','empty.png'):
            trace=self.detect(cv2.imread(str(folder/name)),hints,reference)
            self.assertFalse(any(c['group']=='model_pale' for c in trace['candidates']))

    def test_local_boundary_survives_opposite_edge_brightness_cancellation(self):
        hints=json.loads((FIX/'hints.json').read_text())[-1]['full']
        for name in ('live_boundary.png','live_thin_glare.png'):
            trace=self.detect(cv2.imread(str(FIX/name)),hints)
            self.assertEqual(sum(c['group']=='model_pale' for c in trace['candidates']),4)

    def test_touching_white_parts_are_not_cut_into_artificial_safe_boxes(self):
        reference=np.full((480,640,3),180,np.uint8); frame=reference.copy()
        cv2.rectangle(frame,(200,150),(290,240),(190,190,170),-1)
        cv2.rectangle(frame,(290,150),(380,240),(190,190,170),-1)
        trace=self.detect(frame,[(200,150,291,241,.9),(290,150,381,241,.9)],reference)
        self.assertFalse(any(c['group']=='model_pale' for c in trace['candidates']))
        self.assertTrue(any(c['reason']=='pale_model_component_ambiguous' for c in trace['blocking_regions']))

    def test_low_confidence_or_invalid_hint_never_authorizes_new_part(self):
        frame=cv2.imread(str(FIX/'live2.png'))
        for hints in ([],[(250,50,350,145,.1)],[(250,50,float('nan'),145,.9)]):
            trace=self.detect(frame,hints)
            self.assertFalse(any(c['group']=='model_pale' for c in trace['candidates']))

    def test_existing_gray_component_is_not_duplicated(self):
        reference=np.full((480,640,3),180,np.uint8)
        for color in ((190,190,150),(180,190,140)):
            frame=reference.copy()
            cv2.rectangle(frame,(200,150),(260,210),color,-1)
            trace=self.detect(frame,[(200,150,261,211,.9)],reference)
            self.assertEqual(len(trace['candidates']),1)
            self.assertEqual(trace['candidates'][0]['group'],'model_pale')

    def test_white_components_clipped_by_any_frame_edge_are_obstacles(self):
        reference=np.full((480,640,3),180,np.uint8)
        for box in ((200,0,260,50),(200,430,260,480),(0,150,60,210),(580,150,640,210)):
            frame=reference.copy();x1,y1,x2,y2=box
            cv2.rectangle(frame,(x1,y1),(x2,y2),(190,190,170),-1)
            trace=self.detect(frame,[box+(.9,)],reference)
            self.assertFalse(any(c['group']=='model_pale' for c in trace['candidates']))
            self.assertTrue(any(c['group']=='model_pale' for c in trace['blocking_regions']))

    def test_red_cable_with_a_false_model_hint_stays_excluded(self):
        folder=FIX.parent/'tray_cable_20261117'
        reference=cv2.imread(str(folder/'reference.png'))
        frame=cv2.imread(str(folder/'live_0.png'))
        trace=self.detect(frame,[(0,0,150,480,.99)],reference)
        self.assertFalse(any(c['group']=='model_pale' for c in trace['candidates']))


if __name__=='__main__': unittest.main()
