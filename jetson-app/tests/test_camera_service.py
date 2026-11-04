import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camera_service import CameraError, CameraService, CameraSource


class FakeCapture:
    def __init__(self, opened=True, reads=(), release_error=None):
        self.opened = opened
        self.reads = iter(reads)
        self.release_error = release_error
        self.releases = 0

    def isOpened(self):
        if isinstance(self.opened, Exception):
            raise self.opened
        return self.opened

    def read(self):
        result = next(self.reads, (False, None))
        if isinstance(result, Exception):
            raise result
        return result

    def release(self):
        self.releases += 1
        if self.release_error is not None:
            raise self.release_error


class CameraServiceTests(unittest.TestCase):
    def test_only_requested_sources_open_with_explicit_paths(self):
        left = '/dev/v4l/by-id/usb-left-video-index0'
        right = '/dev/v4l/by-path/pci-usb-right-video-index0'
        top = Path('recordings') / 'top.mp4'
        calls = []

        def factory(source):
            calls.append(source)
            return FakeCapture()

        service = CameraService({'left': left, 'right': right, 'top': top}, factory)
        self.assertEqual(calls, [])
        service.open('top')
        service.open('right')
        service.open('right')
        service.open('left')
        self.assertEqual(calls, [str(top), right, left])
        service.close()

    def test_invalid_configuration_never_opens_a_capture(self):
        configurations = [
            [('left', 'valid.mp4')],
            {'left': 'valid.mp4', 'rear': 'invalid.mp4'},
            {'left': 'valid.mp4', 'top': ''},
            {'left': 0},
            {'left': '/dev/video0'},
            {'left': '/dev/v4l/by-id/'},
            {'left': '/dev/v4l/by-id/../video0'},
        ]
        for sources in configurations:
            with self.subTest(sources=sources):
                with self.assertRaises(CameraError):
                    CameraService(sources, self.fail)

    def test_unconfigured_role_reports_error(self):
        service = CameraService({'left': 'left.mp4'}, self.fail)
        for operation in (service.open, service.read, service.close):
            with self.subTest(operation=operation.__name__):
                with self.assertRaisesRegex(CameraError, 'top'):
                    operation('top')

    def test_open_failure_releases_candidate_and_leaves_other_role_usable(self):
        frame = np.ones((2, 3, 3), dtype=np.uint8)
        for reason in (False, OSError('probe failure')):
            with self.subTest(reason=reason):
                good = FakeCapture(reads=[(True, frame), (True, frame)])
                bad = FakeCapture(opened=reason)
                captures = {'left.mp4': good, 'top.mp4': bad}
                service = CameraService(
                    {'left': 'left.mp4', 'top': 'top.mp4'}, captures.__getitem__
                )
                service.open('left')
                with self.assertRaisesRegex(CameraError, r'top.*top\.mp4'):
                    service.open('top')
                self.assertEqual(bad.releases, 1)
                self.assertEqual(good.releases, 0)
                np.testing.assert_array_equal(service.read('left'), frame)
                self.assertIsNone(service.read('top'))
                service.close('top')
                np.testing.assert_array_equal(service.read('left'), frame)
                service.close()
                self.assertEqual(good.releases, 1)

    def test_close_all_attempts_each_release_even_if_one_raises(self):
        left = FakeCapture(release_error=OSError('release failure'))
        top = FakeCapture()
        captures = {'left.mp4': left, 'top.mp4': top}
        service = CameraService({'left': 'left.mp4', 'top': 'top.mp4'}, captures.__getitem__)
        service.open('left')
        service.open('top')
        with self.assertRaisesRegex(CameraError, r'left.*left\.mp4'):
            service.close()
        service.close()
        self.assertEqual((left.releases, top.releases), (1, 1))


class CameraSourceTests(unittest.TestCase):
    def test_read_returns_independently_owned_contiguous_bgr(self):
        capture_frame = np.full((2, 6, 3), (11, 22, 33), dtype=np.uint8)[:, ::2]
        capture = FakeCapture(reads=[(True, capture_frame), (True, capture_frame)])
        source = CameraSource('sample.mp4', lambda path: capture)
        source.open()
        first = source.read()
        capture_frame.fill(44)
        second = source.read()

        self.assertTrue(first.flags.c_contiguous)
        self.assertEqual(first.shape, (2, 3, 3))
        self.assertEqual(first[0, 0].tolist(), [11, 22, 33])
        self.assertEqual(second[0, 0].tolist(), [44, 44, 44])
        second.fill(0)
        self.assertEqual(capture_frame[0, 0].tolist(), [44, 44, 44])
        source.close()

    def test_bad_frames_return_none_and_later_read_can_recover(self):
        frame = np.zeros((2, 3, 3), dtype=np.uint8)
        capture = FakeCapture(reads=[
            (False, frame),
            (True, None),
            (True, np.zeros((2, 3), dtype=np.uint8)),
            (True, np.zeros((0, 3, 3), dtype=np.uint8)),
            OSError('intermittent read error'),
            (True, frame),
        ])
        source = CameraSource('sample.mp4', lambda path: capture)
        self.assertIsNone(source.read())
        source.open()
        for _ in range(5):
            self.assertIsNone(source.read())
        np.testing.assert_array_equal(source.read(), frame)
        self.assertIsNone(source.read())
        source.close()
        self.assertIsNone(source.read())


if __name__ == '__main__':
    unittest.main()
