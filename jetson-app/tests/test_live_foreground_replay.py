"""Replay real top images through the actual queued reference detector."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from app import QApplication
import cv2
import numpy as np
import linux_20261107 as release
from live_vision import LiveVisionWorker, _confirmed_warm_hints, locate_tray_frame
from motion_limits import MACHINE_REGION, position_in_machine
from tray_background import TrayBackgroundReference
from xy_calibration import CalibrationPoint, XYCalibration

APP=QApplication.instance() or QApplication([])
FIXTURES=Path(__file__).parent/'fixtures/tray_foreground'
SOURCES={role:'/recorded/'+role for role in ('left','right','top')}


class Capture:
    def __init__(self,frame):self.frame=frame
    def isOpened(self):return True
    def read(self):return True,self.frame.copy()
    def release(self):pass


class RealForegroundWorkerReplayTests(unittest.TestCase):
    def test_current_layout_uses_25mm_contact_planning_after_worker_restart(self):
        folder=FIXTURES.parent/'tray_glare_20261116'
        reference=cv2.imread(str(folder/'reference.png'))
        frame=cv2.imread(str(folder/'seven_sequence_7.png'))
        for restart in range(2):
            with self.subTest(restart=restart):
                target,trace=self.run_images(reference,frame)
                self.assertIsNotNone(target)
                self.assertGreaterEqual(target[1],25)
                self.assertEqual(trace['contact_planning']['width_mm'],25)
                self.assertLess(trace['candidates'][trace['selected_index']]['box_full'][0],100)

    def run_case(self,name):
        reference=cv2.imread(str(FIXTURES/'empty98.png'))
        frame=cv2.imread(str(FIXTURES/(name+'.png')))
        return self.run_images(reference,frame)

    def run_images(self,reference,frame,hints=()):
        data=json.loads((Path(__file__).parent/'fixtures/tray_20261116_push_trial.json').read_text(encoding='utf8'))
        calibration=XYCalibration.fit([CalibrationPoint(*p) for p in data['calibration_points']],
                                      pixel_region=(0,0,640,480),physical_region=MACHINE_REGION)
        def unexpected_model():self.fail('Reference localization must not call the side-view recognizer')
        backend=MagicMock()
        worker=LiveVisionWorker(SOURCES,capture_factory=lambda _:Capture(frame),
            tray_factory=lambda:backend,recognizer_factory=unexpected_model,
            tray_compensation=release.TRAY_COMPENSATION,drain_batches=1)
        events,errors=[],[]
        worker.location_evidence_ready.connect(lambda token,evidence:events.append(('evidence',token,evidence)))
        worker.location_ready.connect(lambda token,target:(events.append(('target',token,target)),worker.shutdown()))
        worker.failed.connect(lambda token,error:(errors.append(error),worker.shutdown()))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'tray_background.npz'
            TrayBackgroundReference(reference,SOURCES['top'],release.TRAY_FOREGROUND_ROI).save(path)
            self.assertTrue(worker.locate(41,calibration,release.TRAY_FOREGROUND_ROI,release.PUSH_OFFSET,
                confidence=.75,column_boundaries=(95,152),background_reference=path,motion_target_only=True))
            with patch('live_vision.TrayDetector') as model:
                model.return_value.detect.return_value=hints
                worker.run()
                model.assert_called_once_with(backend,confidence=.4,feature_input=worker.tray_feature_input)
                np.testing.assert_array_equal(model.return_value.detect.call_args[0][0],frame)
        backend.close.assert_called_once_with()
        self.assertEqual(errors,[])
        self.assertEqual([e[0] for e in events],['evidence','target'])
        self.assertTrue(all(e[1]==41 for e in events))
        evidence=events[0][2]
        np.testing.assert_array_equal(evidence['captures']['top'],frame)
        self.assertEqual(evidence['detector_kind'],'foreground')
        return events[1][2],evidence['locator']

    def test_fresh_white_parts_survive_actual_worker_and_25mm_planner(self):
        folder=FIXTURES.parent/'tray_white_20261117'
        hints=json.loads((folder/'hints.json').read_text())[-1]['full']
        target,trace=self.run_images(cv2.imread(str(folder/'reference.png')),
                                     cv2.imread(str(folder/'live2.png')),hints)
        self.assertIsNotNone(target)
        self.assertTrue(position_in_machine(*target))
        self.assertEqual(trace['contact_planning']['width_mm'],25)
        chosen=trace['candidates'][trace['selected_index']]
        self.assertLess(chosen['box_full'][0],350)
        self.assertLess(chosen['box_full'][1],200)

    def test_real_gray_at_left_side_is_found_despite_old_model_low_confidence(self):
        target,trace=self.run_case('gray81')
        self.assertIsNotNone(target)
        self.assertTrue(position_in_machine(*target))
        selected=trace['candidates'][trace['selected_index']]
        self.assertLess(selected['center_mm'][0],95)
        self.assertGreaterEqual(selected['box_full'][2],307)
        self.assertEqual(trace['selection_status'],'selected')

    def test_occluded_front_part_does_not_use_truncated_box_or_skip_to_rear(self):
        target,trace=self.run_case('first36')
        self.assertIsNone(target)
        self.assertEqual(trace['selection_status'],'blocked_earlier_candidate')
        blocked=trace['candidates'][trace['blocked_index']]
        self.assertEqual(blocked['rejection_stage'],'foreground_ambiguity')
        self.assertIn(blocked['reason'],('reference_mask_truncates_part','occluded_or_split_part'))
        self.assertLessEqual(blocked['box_full'][1],30)
        self.assertGreaterEqual(blocked['box_full'][2],218)
        self.assertIn('移开遮挡',trace['hold_reason'])

    def test_independent_real_empty_frame_generates_no_push_target(self):
        target,trace=self.run_case('empty99')
        self.assertIsNone(target)
        self.assertEqual(trace['candidates'],[])
        self.assertIsNone(trace['hold_reason'])

    def test_warm_recheck_runs_on_the_same_frame_before_using_a_boundary_target(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        locator = MagicMock(roi=(16, 4, 624, 476))
        locator.last_diagnostics = {'selection_status': 'selected_forced_boundary'}
        locator.detector.last_diagnostics = {'blocking_regions': [{
            'group': 'warm', 'reason': 'oversized_or_possible_joined_parts',
            'box_roi': (280, 181, 621, 444),
        }]}
        previous = (193.5, 86.0)
        refined = (140.0, 60.0)
        locator.locate.side_effect = [previous, refined]
        with patch('live_vision._confirmed_warm_hints', return_value=[(375, 189, 444, 269, .48)]) as recheck:
            self.assertEqual(locate_tray_frame(locator, frame, None, (), True), refined)
        recheck.assert_called_once_with(None, frame, (), [(296, 185, 637, 448)], True)
        self.assertEqual(locator.detector.confirmed_warm_hints, ((359, 185, 428, 265, .48),))
        self.assertEqual(locator.locate.call_count, 2)
        self.assertTrue(all(args[0][0] is frame for args in locator.locate.call_args_list))

        locator.locate.reset_mock(side_effect=True)
        locator.locate.return_value = previous
        with patch('live_vision._confirmed_warm_hints', return_value=[]) as recheck:
            self.assertEqual(locate_tray_frame(locator, frame, None, (), True), previous)
        recheck.assert_called_once()
        locator.locate.assert_called_once_with(frame)

    def test_weak_recheck_box_requires_two_views_at_the_same_original_coordinates(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        regions = [(296, 185, 637, 448)]
        rotated = [(188.625, 195.5625, 269.375, 264.9375, .482)]
        flipped = [(201.75, 194, 267.75, 272, .26)]
        for second, expected_count in ((flipped, 1), ([], 0)):
            with self.subTest(second=second), patch('live_vision.TrayDetector') as model:
                model.return_value.detect.side_effect = (rotated, second)
                result = _confirmed_warm_hints(None, frame, (), regions, True)
                self.assertEqual(len(result), expected_count)
                if result:
                    np.testing.assert_allclose(result[0], (375.0625, 188.625, 444.4375, 269.375, .482))

    def test_unmatched_reference_is_reported_as_hold_through_actual_worker(self):
        reference=np.full((480,640,3),200,np.uint8)
        target,trace=self.run_images(reference,np.zeros_like(reference))
        self.assertIsNone(target)
        self.assertEqual(trace['selection_status'],'blocked_foreground_reference')
        self.assertIn('底板参考无法稳定匹配',trace['hold_reason'])

    def test_ambiguous_front_region_blocks_later_part_through_actual_worker(self):
        reference=np.full((480,640,3),200,np.uint8);frame=reference.copy()
        cv2.rectangle(frame,(120,120),(420,260),(30,160,30),-1)
        cv2.rectangle(frame,(500,300),(550,350),(180,80,180),-1)
        target,trace=self.run_images(reference,frame)
        self.assertIsNone(target)
        self.assertEqual(trace['selection_status'],'blocked_earlier_candidate')
        blocked=trace['candidates'][trace['blocked_index']]
        self.assertEqual(blocked['rejection_stage'],'foreground_ambiguity')
        self.assertTrue(any(c.get('target_mm') for c in trace['candidates']))


if __name__=='__main__':unittest.main()
