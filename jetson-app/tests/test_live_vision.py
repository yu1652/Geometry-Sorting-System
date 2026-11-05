import threading
import unittest
from types import SimpleNamespace

import numpy as np
from live_vision import LiveVisionWorker


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
    def worker(self, *, shape=(480, 640, 3), fail_at=None, recognizer_factory=None, tray_factory=None):
        events, captures = [], {}

        def capture_factory(path):
            role = path.rsplit('/', 1)[-1]
            captures[role] = Capture(role, events, shape=shape, fail_at=fail_at)
            return captures[role]

        worker = LiveVisionWorker(SOURCES, capture_factory=capture_factory,
                                  recognizer_factory=recognizer_factory,
                                  tray_factory=tray_factory)
        results, failures, previews = [], [], []
        worker.recognition_ready.connect(lambda token, result: (results.append((token, result)), worker.shutdown()))
        worker.location_ready.connect(lambda token, result: (results.append((token, result)), worker.shutdown()))
        worker.failed.connect(lambda token, error: (failures.append((token, error)), worker.shutdown()))
        worker.frames_ready.connect(previews.append)
        return worker, events, captures, results, failures, previews

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
        backends = []

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
            backend = Backend()
            backends.append(backend)
            return backend

        worker, _, _, results, failures, _ = self.worker(tray_factory=factory)
        worker.locate(9, Calibration(), (10, 20, 20, 20), (4, 3))
        self.assertEqual(backends, [])
        worker.run()
        self.assertEqual(failures, [])
        self.assertEqual(results[0][0], 9)
        np.testing.assert_allclose(results[0][1], (26, 33))
        self.assertTrue(backends[0].closed)
        self.assertEqual(backends[0].tensor.shape, (1, 3, 640, 640))

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
