"""Operator-selected relaxation: unresolved regions stop vetoing a proven lane.

Located parts must keep blocking. The relaxed decision has to stay visible in
the location trace, because it trades a hard obstacle for a measured lane.
"""
import unittest

import numpy as np

from tray_locator import TrayLocator
from xy_calibration import XYCalibration

IDENTITY = XYCalibration(np.array(((0, 0), (1, 0), (0, 1), (0, 0), (0, 0), (0, 0)), float),
                         (0, 0, 260, 230), (0, 0, 260, 230))
FRAME = np.zeros((230, 260, 3), np.uint8)
COLUMNS = (95, 152)


def footprint(box):
    x1, y1, x2, y2 = box
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


class Detector:
    def __init__(self, boxes, regions=()):
        self.boxes = [list(box) for box in boxes]
        self.last_diagnostics = {
            'candidates': [{'box_roi': box, 'footprint_roi': footprint(box)} for box in self.boxes],
            'blocking_regions': [{'box_roi': list(region),
                                  'reason': 'occluded_or_split_part'} for region in regions]}
        self.diagnostics = self.last_diagnostics

    def detect(self, frame):
        return [box + [1.0] for box in self.boxes]


def locator(detector, relaxed):
    return TrayLocator(IDENTITY, detector, (0, 0, 260, 230), offsets=(1, 0),
                       motion_target_only=True, column_boundaries=COLUMNS,
                       pusher_width_mm=25, contact_ignore_unresolved=relaxed)


class ContactRelaxationTests(unittest.TestCase):
    # The region grazes the part's lower edge and lies across its whole lane,
    # which is the shape of the board failure recorded on 2026-11-17.
    PART = (60, 30, 100, 70)
    REGION = (10, 60, 80, 110)

    def test_unresolved_region_in_lane_blocks_by_default(self):
        engine = locator(Detector([self.PART], [self.REGION]), False)
        self.assertIsNone(engine.locate(FRAME))
        trace = engine.last_diagnostics
        self.assertEqual(trace['selection_status'], 'blocked_contact_clearance')
        self.assertIn('无碰撞通道', trace['hold_reason'])
        self.assertNotIn('ignored_unresolved_regions', trace['contact_planning'])

    def test_relaxed_lane_clears_when_only_the_unresolved_region_blocks(self):
        engine = locator(Detector([self.PART], [self.REGION]), True)
        target = engine.locate(FRAME)
        self.assertEqual(target, (101.0, 50.0))
        trace = engine.last_diagnostics
        self.assertEqual(trace['selection_status'], 'selected')
        self.assertEqual(trace['contact_planning']['ignored_unresolved_regions'], [1])
        chosen = trace['candidates'][trace['selected_index']]
        self.assertEqual(chosen['relied_on_ignored_regions'], [1])
        self.assertEqual(trace['contact_planning']['selected_relied_on_ignored_regions'], [1])

    def test_relaxed_lane_reports_no_reliance_when_the_region_is_clear(self):
        engine = locator(Detector([self.PART], [(150, 150, 200, 200)]), True)
        self.assertEqual(engine.locate(FRAME), (101.0, 50.0))
        trace = engine.last_diagnostics
        self.assertEqual(trace['contact_planning']['ignored_unresolved_regions'], [1])
        self.assertNotIn('selected_relied_on_ignored_regions', trace['contact_planning'])

    def test_located_part_still_blocks_a_relaxed_lane(self):
        # D sits on the tray edge, so it can never be pushed itself, but it is
        # a located part inside A's object sweep and must keep blocking A. The
        # only stroke left is the blind region fallback, never A's lane.
        engine = locator(Detector([self.PART, (2, 30, 9, 70)], [(150, 150, 200, 200)]), True)
        target = engine.locate(FRAME)
        trace = engine.last_diagnostics
        self.assertEqual(trace['selection_status'], 'selected_blind_region')
        self.assertEqual(trace['selected_index'], 2)
        self.assertTrue(trace['candidates'][2]['blind_region_push'])
        self.assertFalse(trace['candidates'][0].get('selected'))
        self.assertNotEqual(target, (101.0, 50.0))

    def test_unresolved_region_ahead_in_order_blocks_by_default(self):
        # The region's earliest corner sorts before the part but never touches
        # the lane, so only the ordering guard can reject this push.
        engine = locator(Detector([self.PART], [(0, 10, 40, 25)]), False)
        self.assertIsNone(engine.locate(FRAME))
        self.assertEqual(engine.last_diagnostics['selection_status'], 'blocked_earlier_candidate')
        self.assertEqual(engine.last_diagnostics['blocked_index'], 1)

    def test_relaxed_order_planning_keeps_the_region_in_the_trace(self):
        engine = locator(Detector([self.PART], [(0, 10, 40, 25)]), True)
        self.assertEqual(engine.locate(FRAME), (101.0, 50.0))
        trace = engine.last_diagnostics
        self.assertEqual(trace['selection_status'], 'selected')
        self.assertEqual(trace['contact_planning']['earlier_unresolved_index'], 1)
        self.assertNotIn('selected_relied_on_ignored_regions', trace['contact_planning'])

    def test_region_only_scene_pushes_blindly_only_when_relaxed(self):
        self.assertIsNone(locator(Detector([], [self.REGION]), False).locate(FRAME))
        engine = locator(Detector([], [self.REGION]), True)
        target = engine.locate(FRAME)
        self.assertIsNotNone(target)
        trace = engine.last_diagnostics
        self.assertEqual(trace['selection_status'], 'selected_blind_region')
        self.assertEqual(trace['contact_planning']['blind_region_push'], 0)
        self.assertTrue(trace['candidates'][0]['blind_region_push'])

    def test_relaxation_never_authorizes_plain_motion_validation(self):
        with self.assertRaises(ValueError):
            TrayLocator(IDENTITY, Detector([self.PART]), (0, 0, 260, 230),
                        contact_ignore_unresolved=True)
        with self.assertRaises(ValueError):
            TrayLocator(IDENTITY, Detector([self.PART]), (0, 0, 260, 230),
                        motion_target_only=True, contact_ignore_unresolved=1)


if __name__ == '__main__':
    unittest.main()
