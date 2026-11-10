import threading
import unittest
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from live_vision import LiveVisionWorker, open_live_capture
from camera_service import CameraError
import cv2


SOURCES = {role: '/dev/v4l/by-path/' + role for role in ('left', 'right', 'top')}


class Capture:
    def __init__(self, role, events, *, shape=(480, 640, 3), fail_at=None):
        self.role, self.events = role, events
        self.shape, self.fail_at = shape, fail_at
        self.read_count = 0
        self.close_count = 0

    def isOpened(self):
        return True

    def read(self):
        self.read_count += 1
        self.events.append(('read', self.role, self.read_count))
        if self.fail_at is not None and self.read_count >= self.fail_at:
            return False, None
        base = {'left': 20, 'right': 80, 'top': 140}[self.role]
        return True, np.full(self.shape, base + self.read_count, np.uint8)

    def release(self):
        self.close_count += 1
        self.events.append(('close_camera', self.role))


class Recognizer:
    def __init__(self, events):
        self.events = events
        self.owner = threading.get_ident()
        self.close_thread = None

    def recognize_pair(self, left, right):
        self.events.append(('recognize', int(left[0, 0, 0]), int(right[0, 0, 0])))
        return SimpleNamespace(error=None, color=None, detection=None)

    def close(self):
        self.close_thread = threading.get_ident()
        self.events.append(('close_model',))


class Calibration:
    def map_pixel(self, x, y):
        return float(x), float(y)

    def validate_physical(self, x, y):
        if not 0 <= x <= 640 or not 0 <= y <= 480:
            raise ValueError('outside')


class LiveVisionTests(unittest.TestCase):
    def test_warm_models_reports_ready_before_any_motion_job(self):
        calls = []

        class Tray:
            def infer(self, tensor):
                calls.append(('tray', tensor.shape))
                return np.zeros((1, 5, 8400), np.float32)

            def close(self):
                calls.append(('close_tray',))

        worker, _, _, results, failures, previews = self.worker(
            recognizer_factory=lambda: Recognizer(calls), tray_factory=Tray,
            warm_models=True, tray_feature_input=True)
        worker.frames_ready.connect(lambda _: worker.shutdown() if len(previews) >= 10 else None)
        worker.models_ready.connect(lambda: (calls.append(('ready',)), worker.shutdown()))
        worker.run()
        self.assertEqual([call[0] for call in calls], ['recognize', 'tray', 'ready', 'close_model', 'close_tray'])
        self.assertEqual(results, [])
        self.assertEqual(failures, [])

    def test_camera_fps_request_is_optional_when_driver_keeps_30_fps(self):
        capture = MagicMock()
        capture.isOpened.return_value = True
        capture.set.side_effect = lambda prop, value: prop != cv2.CAP_PROP_FPS
        mjpg = cv2.VideoWriter_fourcc(*'MJPG')
        capture.get.side_effect = lambda prop: mjpg if prop == cv2.CAP_PROP_FOURCC else 30.0
        notices = []
        with patch('live_vision.cv2.VideoCapture', return_value=capture):
            result = open_live_capture('/dev/v4l/by-path/top', notices.append)
        self.assertIs(result, capture)
        capture.release.assert_not_called()
        self.assertTrue(any('30' in line for line in notices))

    def test_unreported_fps_does_not_reject_valid_mjpg_capture(self):
        for fps in (0.0, float('nan')):
            with self.subTest(fps=fps):
                capture = MagicMock()
                capture.isOpened.return_value = True
                capture.set.return_value = True
                mjpg = cv2.VideoWriter_fourcc(*'MJPG')
                capture.get.side_effect = lambda prop: mjpg if prop == cv2.CAP_PROP_FOURCC else fps
                notices = []
                with patch('live_vision.cv2.VideoCapture', return_value=capture):
                    self.assertIs(open_live_capture('/dev/v4l/by-path/top', notices.append), capture)
                self.assertTrue(any('未知' in line for line in notices))

    def test_non_mjpg_capture_is_still_rejected_and_released(self):
        capture = MagicMock()
        capture.isOpened.return_value = True
        capture.set.return_value = True
        capture.get.return_value = cv2.VideoWriter_fourcc(*'YUYV')
        with patch('live_vision.cv2.VideoCapture', return_value=capture):
            with self.assertRaisesRegex(CameraError, 'MJPG'):
                open_live_capture('/dev/v4l/by-path/top')
        capture.release.assert_called_once()

    def worker(self, *, shape=(480, 640, 3), fail_at=None, recognizer_factory=None, tray_factory=None, **settings):
        events, captures = [], {}

        def capture_factory(path):
            role = path.rsplit('/', 1)[-1]
            captures[role] = Capture(role, events, shape=shape, fail_at=fail_at)
            return captures[role]

        worker = LiveVisionWorker(SOURCES, capture_factory=capture_factory,
                                  recognizer_factory=recognizer_factory,
                                  tray_factory=tray_factory, **settings)
        results, failures, previews = [], [], []
        worker.recognition_ready.connect(lambda token, result: (results.append((token, result)), worker.shutdown()))
        worker.location_ready.connect(lambda token, result: (results.append((token, result)), worker.shutdown()))
        worker.failed.connect(lambda token, error: (failures.append((token, error)), worker.shutdown()))
        worker.frames_ready.connect(previews.append)
        return worker, events, captures, results, failures, previews

    def test_empty_reference_settles_drains_copies_all_views_without_loading_models(self):
        for preview_seconds in (.1, 1.0):
            with self.subTest(preview_seconds=preview_seconds):
                now, settled_from, references = [0.0], [], []
                worker, _, captures, results, failures, previews = self.worker(clock=lambda: now[0])
                worker.recognizer_factory = lambda: self.fail('empty reference must not load recognizer')
                worker.tray_factory = lambda: self.fail('empty reference must not load tray model')

                def preview(frames):
                    now[0] += preview_seconds
                    if len(previews) == 2:
                        settled_from.append(now[0])

                worker.frames_ready.connect(preview)
                worker.empty_reference_ready.connect(
                    lambda token, frames: (references.append((token, frames)), worker.shutdown()))
                self.assertTrue(worker.capture_empty(37))
                worker.run()
                self.assertEqual(failures, [])
                self.assertEqual(results, [])
                self.assertEqual(references[0][0], 37)
                self.assertGreaterEqual(now[0] - settled_from[0], .8)
                self.assertTrue(all(cap.read_count >= 8 for cap in captures.values()))
                self.assertEqual(set(references[0][1]), {'left', 'right', 'top'})
                for role, frame in references[0][1].items():
                    np.testing.assert_array_equal(frame, previews[-1][role])
                    self.assertFalse(np.shares_memory(frame, previews[-1][role]))
                    original = frame.copy()
                    previews[-1][role].fill(0)
                    np.testing.assert_array_equal(frame, original)
                self.assertTrue(all(cap.close_count == 1 for cap in captures.values()))

    def test_empty_reference_shutdown_during_settling_emits_no_old_reference(self):
        now, references = [0.0], []
        worker, _, captures, results, failures, previews = self.worker(clock=lambda: now[0])
        worker.empty_reference_ready.connect(lambda token, frames: references.append((token, frames)))

        def preview(frames):
            now[0] += .1
            if len(previews) == 4:
                worker.shutdown()

        worker.frames_ready.connect(preview)
        self.assertTrue(worker.capture_empty(38))
        worker.run()
        self.assertEqual(references, [])
        self.assertEqual(results, [])
        self.assertEqual(failures, [])
        self.assertTrue(all(cap.close_count == 1 for cap in captures.values()))

    def test_empty_reference_camera_failure_during_drain_never_returns_preview(self):
        references = []
        worker, _, captures, results, failures, previews = self.worker(fail_at=6)
        worker.empty_reference_ready.connect(lambda token, frames: references.append((token, frames)))
        self.assertTrue(worker.capture_empty(39))
        worker.run()
        self.assertTrue(previews)
        self.assertEqual(references, [])
        self.assertEqual(results, [])
        self.assertEqual(failures[0][0], -1)
        self.assertIn('fresh frame', failures[0][1])
        self.assertTrue(all(cap.close_count == 1 for cap in captures.values()))

    def test_manual_recognition_discards_frames_captured_before_lazy_model_load(self):
        worker, events, captures, results, failures, _ = self.worker()
        loaded_counts = {}

        def load():
            loaded_counts.update({role: cap.read_count for role, cap in captures.items()})
            return Recognizer(events)

        worker.recognizer_factory = load
        worker.recognize(13)
        worker.run()
        self.assertEqual(failures, [])
        self.assertEqual(results[0][0], 13)
        call = next(event for event in events if event[0] == 'recognize')
        self.assertGreaterEqual(call[1] - 20, loaded_counts['left'] + 4)
        self.assertGreaterEqual(call[2] - 80, loaded_counts['right'] + 4)

    def test_automatic_recognition_keeps_previewing_until_object_arrives_after_load(self):
        now, loaded_at, calls = [0.0], [], []
        worker, _, captures, results, failures, previews = self.worker(clock=lambda: now[0])

        class Model:
            def recognize_pair(self, left, right):
                calls.append((now[0], int(left[0, 0, 0]), int(right[0, 0, 0])))
                return SimpleNamespace(error=None, diagnostics={})

            def close(self):
                pass

        def load():
            now[0] += 2.0  # TensorRT initialization is slow.
            loaded_at.append(now[0])
            for role, capture in captures.items():
                old_read = capture.read

                def read(old_read=old_read, role=role):
                    ok, frame = old_read()
                    frame.fill(200 if now[0] >= loaded_at[0] + .6 else 0)
                    if role == 'top':
                        now[0] += .1
                    return ok, frame

                capture.read = read
            return Model()

        worker.recognizer_factory = load
        self.assertTrue(worker.recognize(14, settle=True))
        worker.run()
        self.assertEqual(failures, [])
        self.assertEqual(results[0][0], 14)
        self.assertGreaterEqual(calls[0][0] - loaded_at[0], .8)
        self.assertEqual(calls[0][1:], (200, 200))
        self.assertGreaterEqual(len(previews), 9)
        timing = results[0][1].diagnostics['capture']
        self.assertTrue(timing['settle_requested'])
        self.assertEqual(timing['settle_seconds'], .8)
        self.assertGreaterEqual(timing['capture_completed_monotonic'] - loaded_at[0], .8)
        self.assertEqual(timing['inference_completed_monotonic'], calls[0][0])

    def test_shutdown_during_settling_never_runs_inference_and_closes_model(self):
        now, loaded, closed, inference = [0.0], [], [], []
        worker, _, captures, results, failures, previews = self.worker(clock=lambda: now[0])

        class Model:
            def recognize_pair(self, left, right):
                inference.append(True)
                return SimpleNamespace(error=None)

            def close(self):
                closed.append(threading.get_ident())

        def load():
            loaded.append(threading.get_ident())
            return Model()

        def preview(frames):
            if loaded:
                now[0] += .1
                if now[0] >= .2:
                    worker.shutdown()

        worker.recognizer_factory = load
        worker.frames_ready.connect(preview)
        worker.recognize(15, settle=True)
        worker.run()
        self.assertEqual(inference, [])
        self.assertEqual(results, [])
        self.assertEqual(failures, [])
        self.assertEqual(closed, loaded)
        self.assertTrue(all(cap.close_count == 1 for cap in captures.values()))

    def test_diagnostic_capture_snapshots_and_timestamps_survive_error_results(self):
        for error in (None, 'No target detected'):
            with self.subTest(error=error):
                now, captured = [10.0], []
                diagnostic = {'model': 'legacy'}
                result = SimpleNamespace(error=error, diagnostics=diagnostic)
                worker, _, _, results, failures, previews = self.worker(clock=lambda: now[0])

                class Model:
                    def recognize_pair(self, left, right):
                        left.fill(0)  # Evidence must survive a mutating backend.
                        now[0] += .2
                        return result

                    def close(self):
                        pass

                def load():
                    now[0] += 2.0
                    return Model()

                def preview(frames):
                    captured[:] = [{role: frame.copy() for role, frame in frames.items()}]
                    now[0] += .1

                worker.recognizer_factory = load
                worker.frames_ready.connect(preview)
                worker.recognize(19)
                worker.run()
                self.assertEqual(failures, [])
                self.assertEqual(results, [(19, result)])
                self.assertEqual(diagnostic['model'], 'legacy')
                self.assertEqual(set(diagnostic['captures']), {'left', 'right', 'top'})
                for role, frame in diagnostic['captures'].items():
                    np.testing.assert_array_equal(frame, captured[0][role])
                    self.assertFalse(np.shares_memory(frame, previews[-1][role]))
                timing = diagnostic['capture']
                self.assertFalse(timing['settle_requested'])
                self.assertEqual(timing['settle_seconds'], 0)
                self.assertEqual(timing['drain_batches'], 4)
                self.assertEqual(timing['request_received_monotonic'], 10.0)
                self.assertAlmostEqual(timing['capture_completed_monotonic'], 12.6)
                self.assertAlmostEqual(timing['inference_completed_monotonic'], 12.8)

    def test_idle_preview_pumps_cameras_without_waiting_for_a_request(self):
        now, preview_times = [0.0], []
        worker, _, captures, _, failures, _ = self.worker(clock=lambda: now[0])

        class Mailbox(Queue):
            def get(self, block=True, timeout=None):
                if block:
                    now[0] += timeout or 1
                return super().get(block=False)

        def preview(frames):
            preview_times.append(now[0])
            if len(preview_times) == 4:
                worker.shutdown()

        worker._requests = Mailbox(maxsize=1)
        worker.frames_ready.connect(preview)
        with patch.object(worker._stopping, 'wait', side_effect=lambda seconds: now.__setitem__(0, now[0] + seconds)):
            worker.run()
        self.assertEqual(failures, [])
        self.assertEqual(len(preview_times), 4)
        self.assertLessEqual(preview_times[-1] - preview_times[0], .02)
        self.assertTrue(all(cap.close_count == 1 for cap in captures.values()))

    def test_invalid_capture_timing_cannot_create_unbounded_or_empty_drain(self):
        for settings in ({'scan_settle_seconds': float('nan')},
                         {'scan_settle_seconds': float('inf')},
                         {'scan_settle_seconds': -1}, {'scan_settle_seconds': 31},
                         {'drain_batches': 0}, {'drain_batches': 101},
                         {'drain_batches': 1.5}, {'drain_batches': True},
                         {'clock': None}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                self.worker(**settings)

    def test_recognition_uses_fresh_raw_frames_in_training_order_and_releases_all(self):
        models = []
        worker, events, captures, results, failures, previews = self.worker()

        def create():
            models.append(Recognizer(events))
            return models[-1]

        worker.recognizer_factory = create
        self.assertTrue(worker.recognize(7))
        worker.run()
        self.assertEqual(failures, [])
        self.assertEqual(results[0][0], 7)
        self.assertGreaterEqual(len(previews), 2)
        call = next(event for event in events if event[0] == 'recognize')
        self.assertEqual(call[1] - 20, call[2] - 80)
        self.assertGreaterEqual(call[1] - 20, 5)
        self.assertGreater(call[1], int(previews[0]['left'][0, 0, 0]))
        self.assertEqual(set(previews[-1]), {'left', 'right', 'top'})
        self.assertTrue(all(capture.close_count == 1 for capture in captures.values()))
        self.assertEqual(models[0].owner, models[0].close_thread)

    def test_location_uses_roi_and_offsets_and_lazily_closes_tray_backend(self):
        backends, load_counts = [], []

        class Backend:
            def __init__(self):
                self.closed = False

            def infer(self, tensor):
                self.tensor = tensor
                raw = np.zeros((1, 5, 8400), np.float32)
                raw[0, :, 0] = (320, 320, 128, 128, .9)
                return raw

            def close(self):
                self.closed = True

        def factory():
            load_counts.append(captures['top'].read_count)
            backend = Backend()
            backends.append(backend)
            return backend

        worker, _, captures, results, failures, _ = self.worker(tray_factory=factory)
        worker.locate(9, Calibration(), (10, 20, 20, 20), (4, 3))
        self.assertEqual(backends, [])
        worker.run()
        self.assertEqual(failures, [])
        self.assertEqual(results[0][0], 9)
        np.testing.assert_allclose(results[0][1], (26, 33))
        self.assertTrue(backends[0].closed)
        self.assertEqual(backends[0].tensor.shape, (1, 3, 640, 640))
        self.assertAlmostEqual(float(backends[0].tensor[0, 0, 320, 320]),
                               (140 + load_counts[0] + 4) / 255, places=6)

    def test_camera_loss_does_not_reuse_last_preview_for_recognition(self):
        worker, _, captures, results, failures, previews = self.worker(fail_at=4)
        worker.recognizer_factory = lambda: self.fail('model must not load with missing camera')
        worker.recognize(2)
        worker.run()
        self.assertEqual(results, [])
        self.assertEqual(failures[0][0], -1)
        self.assertTrue(previews)
        self.assertTrue(all(capture.close_count == 1 for capture in captures.values()))

    def test_invalid_frame_dimensions_are_fatal(self):
        worker, _, _, results, failures, _ = self.worker(shape=(240, 320, 3))
        worker.run()
        self.assertEqual(results, [])
        self.assertEqual(failures[0][0], -1)
        self.assertIn('640', failures[0][1])

    def test_load_error_keeps_request_token(self):
        def failed_loader():
            raise RuntimeError('broken engine')

        worker, _, captures, results, failures, _ = self.worker(recognizer_factory=failed_loader)
        worker.recognize(42)
        worker.run()
        self.assertEqual(results, [])
        self.assertEqual(failures, [(42, 'broken engine')])
        self.assertTrue(all(capture.close_count == 1 for capture in captures.values()))

    def test_shutdown_discards_queued_work_and_request_queue_is_bounded(self):
        worker, _, captures, results, failures, _ = self.worker()
        self.assertTrue(worker.recognize(1))
        self.assertFalse(worker.recognize(2))
        self.assertEqual(failures[0][0], 2)
        worker.shutdown()
        worker.run()
        self.assertEqual(captures, {})
        self.assertEqual(results, [])

    def test_three_distinct_camera_sources_are_required(self):
        with self.assertRaises(ValueError):
            LiveVisionWorker({'left': SOURCES['left']})
        with self.assertRaises(ValueError):
            LiveVisionWorker(dict.fromkeys(SOURCES, SOURCES['left']))

    def test_partial_camera_open_failure_releases_already_open_devices(self):
        captures, failures = [], []

        def factory(path):
            if path == SOURCES['right']:
                raise RuntimeError('USB unavailable')
            capture = Capture('left', [])
            captures.append(capture)
            return capture

        worker = LiveVisionWorker(SOURCES, capture_factory=factory)
        worker.failed.connect(lambda token, error: failures.append((token, error)))
        worker.run()
        self.assertEqual(failures[0][0], -1)
        self.assertIn('USB unavailable', failures[0][1])
        self.assertEqual(captures[0].close_count, 1)

    def test_real_worker_thread_owns_models_and_shutdown_suppresses_late_result(self):
        main_thread = threading.get_ident()
        created_on, closed_on, results = [], [], []

        class Model:
            def __init__(self):
                created_on.append(threading.get_ident())

            def recognize_pair(self, left, right):
                worker.shutdown()
                return SimpleNamespace(error=None)

            def close(self):
                closed_on.append(threading.get_ident())

        worker = LiveVisionWorker(
            SOURCES, capture_factory=lambda path: Capture(path.rsplit('/', 1)[-1], []),
            recognizer_factory=Model,
        )
        worker.recognition_ready.connect(lambda token, result: results.append(token))
        worker.recognize(10)
        worker.start()
        self.assertTrue(worker.wait(3000), 'worker did not shut down')
        self.assertEqual(created_on, closed_on)
        self.assertEqual(len(created_on), 1)
        self.assertNotEqual(created_on[0], main_thread)
        self.assertEqual(results, [])


if __name__ == '__main__':
    unittest.main()
