"""An object's geometric center is not a commanded motor position."""
import json
from pathlib import Path
import unittest
import numpy as np

import linux_20261107 as release
from motion_limits import MACHINE_REGION
from tray_locator import TrayDetector, TrayLocator
from xy_calibration import XYCalibration, CalibrationPoint


class Recorded:
    def __init__(self, boxes): self.boxes = boxes
    def detect(self, frame): return self.boxes


class TrayTargetGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture = json.loads((Path(__file__).parent/'fixtures/tray_20261116_push_trial.json').read_text(encoding='utf8'))
        cls.points = [CalibrationPoint(*row) for row in fixture['calibration_points']]
        cls.frame = np.zeros((480,640,3),np.uint8)

    def calibration(self, region=MACHINE_REGION):
        return XYCalibration.fit(self.points,pixel_region=(16,0,535,479),physical_region=region)

    def locator(self, boxes, **kwargs):
        cal = kwargs.pop('calibration',self.calibration())
        settings = dict(offsets=release.PUSH_OFFSET,confidence=.75,
                        compensation=release.TRAY_COMPENSATION,skip_unsafe=True,
                        column_boundaries=(95,152),motion_target_only=True)
        settings.update(kwargs)
        try:
            return TrayLocator(cal,Recorded(boxes),release.TRAY_ROI,**settings)
        except TypeError as error:
            self.fail('Tray locator must support target-only motion validation: '+str(error))

    def test_edge_center_outside_old_pick_region_keeps_legal_rear_start(self):
        locator = self.locator([(134,230,156,250,.99)])  # full x=150..172
        np.testing.assert_allclose(locator.locate(self.frame),(47.9906,58.5393),atol=.001)
        trace = locator.last_diagnostics
        self.assertEqual(trace['selected_index'],0)
        self.assertLess(trace['candidates'][0]['center_mm'][0],release.PICK_REGION[0])

    def test_negative_geometric_center_y_can_have_legal_compensated_target(self):
        locator = self.locator([(214,90,234,110,.99)])
        np.testing.assert_allclose(locator.locate(self.frame),(81.8936,1.6191),atol=.001)
        self.assertLess(locator.last_diagnostics['candidates'][0]['center_mm'][1],0)

    def test_default_legacy_center_validation_is_preserved(self):
        locator = TrayLocator(self.calibration(release.PICK_REGION),Recorded([(134,230,156,250,.99)]),
            release.TRAY_ROI,offsets=release.PUSH_OFFSET,compensation=release.TRAY_COMPENSATION,skip_unsafe=True)
        self.assertIsNone(locator.locate(self.frame))

    def simple_calibration(self):
        coefficients=np.array([[0,-30],[1,0],[0,1],[0,0],[0,0],[0,0]],float)
        return XYCalibration(coefficients,(16,0,535,479),MACHINE_REGION)

    def test_earlier_unreachable_part_blocks_later_column(self):
        locator=self.locator([(24,0,44,20,.99),(114,100,134,120,.99)],
            calibration=self.simple_calibration(),compensation=None)
        self.assertIsNone(locator.locate(self.frame))
        self.assertEqual(locator.last_diagnostics['selection_status'],'blocked_earlier_candidate')
        self.assertEqual(locator.last_diagnostics['blocked_index'],0)
        self.assertTrue(locator.last_diagnostics['hold_reason'])
        self.assertEqual(locator.last_diagnostics['candidates'][0]['proposed_target_mm'],[61.,-20.])
        self.assertIn('Y=-20.0',locator.last_diagnostics['hold_reason'])

    def test_later_unreachable_part_does_not_block_first_column(self):
        locator=self.locator([(184,0,204,20,.99),(24,100,44,120,.99)],
            calibration=self.simple_calibration(),compensation=None)
        self.assertEqual(locator.locate(self.frame),(61.,80.))
        self.assertEqual(locator.last_diagnostics['selected_index'],1)

    def test_target_past_real_travel_is_rejected_without_clamping(self):
        locator=self.locator([(230,100,260,120,.99)],calibration=self.simple_calibration(),compensation=None)
        self.assertIsNone(locator.locate(self.frame))
        self.assertEqual(locator.last_diagnostics['candidates'][0]['rejection_stage'],'target_range')

    def test_rounded_push_start_must_strictly_exceed_end_x(self):
        coefficients=np.array([[0,0],[.1,0],[0,1],[0,0],[0,0],[0,0]],float)
        cal=XYCalibration(coefficients,(16,0,535,479),MACHINE_REGION)
        locator=self.locator([(44,100,74.004,120,.99)],calibration=cal,compensation=None)
        self.assertIsNone(locator.locate(self.frame))
        self.assertEqual(locator.last_diagnostics['candidates'][0]['rejection_stage'],'target_range')

    def test_nonfinite_and_weak_candidates_are_logged_not_ranked(self):
        locator=self.locator([(24,100,44,120,.2),(24,100,44,120,float('nan')),
                              (114,100,134,120,.99)],calibration=self.simple_calibration(),compensation=None)
        self.assertEqual(locator.locate(self.frame),(151.,80.))
        trace=locator.last_diagnostics
        self.assertEqual(trace['selected_index'],2)
        self.assertEqual([r['rejection_stage'] for r in trace['candidates'][:2]],['confidence','decoded_values'])
        # Diagnostics must remain JSON-serializable without allowing NaN.
        json.dumps(trace,allow_nan=False)

    def test_motion_flag_requires_a_boolean(self):
        for value in (None,0,1,'yes'):
            with self.subTest(value=value):
                try:
                    with self.assertRaises(ValueError):
                        TrayLocator(self.calibration(),Recorded([]),release.TRAY_ROI,motion_target_only=value)
                except TypeError as error:
                    self.fail('Missing motion validation option: '+str(error))

    def test_nonfinite_or_invalid_threshold_cannot_authorize_motion_or_break_trace(self):
        for threshold in (float('nan'),float('inf'),-0.1,1.1,True,None,'0.5'):
            with self.subTest(threshold=threshold):
                with self.assertRaises(ValueError):
                    TrayLocator(self.calibration(),Recorded([]),release.TRAY_ROI,confidence=threshold)
                with self.assertRaises(ValueError):
                    TrayDetector(None,confidence=threshold)

    def ambiguous_locator(self, boxes, diagnostics):
        detector=Recorded(boxes)
        detector.last_diagnostics=diagnostics
        return TrayLocator(self.simple_calibration(),detector,release.TRAY_ROI,
            offsets=(1,0),skip_unsafe=True,column_boundaries=(95,152),motion_target_only=True)

    def test_known_ambiguous_front_region_blocks_later_piece(self):
        locator=self.ambiguous_locator([(174,100,194,120,1.)],
            {'blocking_regions':[{'box_roi':[24,100,144,220],
                                  'reason':'oversized_or_possible_joined_parts'}]})
        self.assertIsNone(locator.locate(self.frame))
        trace=locator.last_diagnostics
        self.assertEqual(trace['selection_status'],'blocked_earlier_candidate')
        self.assertIn('分开',trace['hold_reason'])
        self.assertEqual(trace['candidates'][trace['blocked_index']]['rejection_stage'],'foreground_ambiguity')

    def test_later_ambiguous_region_allows_earlier_separate_piece(self):
        locator=self.ambiguous_locator([(24,100,44,120,1.)],
            {'blocking_regions':[{'box_roi':[184,100,234,220],
                                  'reason':'oversized_or_possible_joined_parts'}]})
        self.assertEqual(locator.locate(self.frame),(61.,80.))

    def test_unstable_reference_never_authorizes_other_detected_parts(self):
        locator=self.ambiguous_locator([(24,100,44,120,1.)],
            {'hold_reason':'底板参考无法稳定匹配，请重新保存底板'})
        self.assertIsNone(locator.locate(self.frame))
        self.assertEqual(locator.last_diagnostics['selection_status'],'blocked_foreground_reference')
        self.assertIn('底板参考',locator.last_diagnostics['hold_reason'])

    def test_malformed_ambiguity_cannot_be_ignored_to_push_later_parts(self):
        locator=self.ambiguous_locator([(174,100,194,120,1.)],
            {'blocking_regions':[{'box_roi':[float('nan'),100,144,220],
                                  'reason':'oversized_or_possible_joined_parts'}]})
        with self.assertRaises(ValueError):
            locator.locate(self.frame)


if __name__=='__main__':unittest.main()
