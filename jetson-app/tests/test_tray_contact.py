import unittest
import numpy as np
import json
from pathlib import Path
import cv2

try:
    from tray_contact import plan_push_contact
except ImportError:
    plan_push_contact=None

def box(x1,y1,x2,y2):
    return np.array(((x1,y1),(x2,y1),(x2,y2),(x1,y2)),float)

class ContactTests(unittest.TestCase):
    def plan(self,part,others=(),preferred=(101,50),region=(0,0,260,230)):
        self.assertIsNotNone(plan_push_contact)
        return plan_push_contact(part,others,preferred,region,width_mm=25,gap_mm=1)

    def test_clear_normal_target_is_unchanged(self):
        self.assertEqual(self.plan(box(60,30,100,70)),(101.,50.))

    def test_edge_center_outside_travel_uses_observed_reachable_contact(self):
        result=self.plan(box(60,-30,100,15),preferred=(101,-8))
        self.assertEqual(result,(101.,0.))

    def test_entire_part_outside_reach_is_not_clamped(self):
        self.assertIsNone(self.plan(box(60,-60,100,-30),preferred=(101,-40)))

    def test_target_beyond_x_travel_stays_rejected(self):
        self.assertIsNone(self.plan(box(250,30,270,70),preferred=(271,50)))

    def test_old_wider_calibration_cannot_expand_x_travel(self):
        self.assertEqual(self.plan(box(170,30,209,70),preferred=(210,50)),(210.,50.))
        self.assertIsNone(self.plan(box(170,30,210,70),preferred=(211,50)))
        self.assertIsNone(self.plan(box(170,30,209,70),preferred=(210.001,50)))

    def test_triangle_contact_must_occur_before_push_endpoint(self):
        triangle=np.array(((0,0),(0,100),(20,100)),float)
        result=self.plan(triangle,preferred=(21,0))
        self.assertIsNotNone(result)
        self.assertGreater(result[1],37.5)

    def test_part_entirely_past_push_endpoint_does_not_get_noop_target(self):
        self.assertIsNone(self.plan(box(0,30,9,70),preferred=(13,50)))

    def test_plate_cannot_descend_on_adjacent_part(self):
        self.assertIsNone(self.plan(box(60,30,100,70),[box(100.5,0,130,100)]))

    def test_object_sweep_blocks_even_if_plate_can_miss_obstacle(self):
        self.assertIsNone(self.plan(box(60,30,100,90),[box(20,80,40,95)]))

    def test_contact_can_shift_away_from_rear_neighbour(self):
        result=self.plan(box(60,30,100,90),[box(101,30,130,55)])
        self.assertIsNotNone(result)
        self.assertGreater(result[1]-12.5,55)

    def test_current_contact_is_allowed_only_when_moving_away(self):
        self.assertIsNotNone(self.plan(box(60,30,100,90),[box(99,30,130,50)],preferred=(101,75)))
        self.assertIsNone(self.plan(box(60,30,100,90),[box(40,70,61,100)],preferred=(101,45)))

    def test_malformed_geometry_and_width_fail_closed(self):
        self.assertIsNotNone(plan_push_contact)
        for part in ([],[[0,0]],[[0,0],[1,1],[float('nan'),2]]):
            with self.assertRaises(ValueError):
                plan_push_contact(part,[],(100,50),(0,0,260,230),width_mm=25,gap_mm=1)


class ContactLocatorTests(unittest.TestCase):
    def test_live_layout_clears_gray_before_blocked_top_blue(self):
        from tray_foreground import TrayForegroundDetector
        from tray_locator import TrayLocator
        from xy_calibration import CalibrationPoint,XYCalibration
        import linux_20261107 as release
        fixtures=Path(__file__).parent/'fixtures'
        data=json.loads((fixtures/'tray_20261116_push_trial.json').read_text(encoding='utf8'))
        calibration=XYCalibration.fit([CalibrationPoint(*p) for p in data['calibration_points']],
            pixel_region=(0,0,640,480),physical_region=(0,0,260,230))
        ref=cv2.imread(str(fixtures/'tray_glare_20261116/reference.png'))
        for path in sorted((fixtures/'tray_glare_20261116').glob('seven_sequence_*.png')):
            with self.subTest(frame=path.name):
                locator=TrayLocator(calibration,TrayForegroundDetector(ref),(0,0,640,480),
                    offsets=(1,0),compensation=release.TRAY_COMPENSATION,skip_unsafe=True,
                    column_boundaries=(95,152),motion_target_only=True,pusher_width_mm=25)
                target=locator.locate(cv2.imread(str(path)))
                self.assertIsNotNone(target)
                self.assertTrue(85<target[0]<100)
                self.assertGreaterEqual(target[1],25)
                trace=locator.last_diagnostics
                chosen=trace['candidates'][trace['selected_index']]
                self.assertLess(chosen['box_full'][0],100)
                self.assertGreater(chosen['contact_adjustment_y_mm'],0)
                self.assertEqual(trace['contact_planning']['width_mm'],25)

    def test_missing_footprint_never_authorizes_off_center_push(self):
        from tray_locator import TrayLocator
        from xy_calibration import XYCalibration
        class Detector:
            last_diagnostics={'candidates':[{'box_roi':[60,30,100,70]}]}
            def detect(self,frame):return [[60,30,100,70,1]]
        cal=XYCalibration(np.array(((0,0),(1,0),(0,1),(0,0),(0,0),(0,0)),float),(0,0,260,230),(0,0,260,230))
        locator=TrayLocator(cal,Detector(),(0,0,260,230),motion_target_only=True,pusher_width_mm=25)
        self.assertIsNone(locator.locate(np.zeros((230,260,3),np.uint8)))
        self.assertEqual(locator.last_diagnostics['selection_status'],'blocked_contact_geometry')
