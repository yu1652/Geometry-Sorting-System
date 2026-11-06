from pathlib import Path
from hashlib import sha256
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scene_presence import EmptySceneReference


def background():
    yy, xx = np.indices((480, 640))
    gray = (95 + (xx % 41) + (yy % 31)).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


class ScenePresenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'empty_scene.npz'
        self.ref = EmptySceneReference(self.path)
        self.frames = {'left': background(), 'right': background()}
        self.sources = {'left': '/dev/video0', 'right': '/dev/video2', 'top': '/dev/video4'}

    def test_reference_and_sensor_noise_are_empty_and_arrays_unchanged(self):
        originals = {role: frame.copy() for role, frame in self.frames.items()}
        self.assertIsNone(self.ref.record(self.frames, self.sources))
        self.assertEqual(self.ref.assess(self.frames, self.sources).state, 'empty')
        noise = np.random.default_rng(2).integers(-2, 3, (480, 640, 3))
        noisy = {role: (frame.astype(np.int16) + noise).clip(0, 255).astype(np.uint8) for role, frame in self.frames.items()}
        result = EmptySceneReference(self.path).assess(noisy, self.sources)
        self.assertEqual(result.state, 'empty')
        self.assertEqual(set(result.scores), {'left', 'right'})
        for role in self.frames:
            np.testing.assert_array_equal(self.frames[role], originals[role])
        self.path.unlink()  # no persistent file handles

    def test_black_white_colored_objects_on_either_view_are_occupied(self):
        self.ref.record(self.frames, self.sources)
        for role in ('left', 'right'):
            for color in ((0, 0, 0), (255, 255, 255), (20, 40, 230)):
                frames = {key: value.copy() for key, value in self.frames.items()}
                frames[role][180:225, 280:325] = color
                with self.subTest(role=role, color=color):
                    self.assertEqual(self.ref.assess(frames, self.sources).state, 'occupied')

    def test_tiny_local_change_cannot_be_called_empty(self):
        self.ref.record(self.frames, self.sources)
        frames = {key: value.copy() for key, value in self.frames.items()}
        frames['left'][200:204, 300:304] = 0
        self.assertNotEqual(self.ref.assess(frames, self.sources).state, 'empty')

    def test_global_light_change_and_bad_frame_are_unknown(self):
        self.ref.record(self.frames, self.sources)
        for delta in (-65, 65):
            frames = {key: (value.astype(np.int16) + delta).clip(0, 255).astype(np.uint8) for key, value in self.frames.items()}
            self.assertEqual(self.ref.assess(frames, self.sources).state, 'unknown')
        for bad in (None, np.zeros((480, 640, 3), np.uint8), np.full((480, 640, 3), 255, np.uint8),
                    np.zeros((20, 20, 3), np.uint8), np.zeros((480, 640, 3), np.float32)):
            frames = dict(self.frames, left=bad)
            self.assertEqual(self.ref.assess(frames, self.sources).state, 'unknown')

    def test_uncalibrated_changed_source_and_corrupt_reference_are_unknown(self):
        self.assertEqual(self.ref.assess(self.frames, self.sources).state, 'unknown')
        self.assertFalse(self.path.exists())
        self.ref.record(self.frames, self.sources)
        swapped = dict(self.sources, left='/dev/video9')
        self.assertEqual(self.ref.assess(self.frames, swapped).state, 'unknown')
        top_changed = dict(self.sources, top='/dev/video99')
        self.assertEqual(self.ref.assess(self.frames, top_changed).state, 'empty')
        self.path.write_bytes(b'corrupt')
        self.assertEqual(EmptySceneReference(self.path).assess(self.frames, self.sources).state, 'unknown')

    def test_record_rejects_invalid_reference_and_preserves_previous_on_write_failure(self):
        self.ref.record(self.frames, self.sources)
        original_file = self.path.read_bytes()
        for value in (0, 255, 120):
            bad = dict(self.frames, left=np.full((480, 640, 3), value, np.uint8))
            with self.assertRaises(ValueError):
                self.ref.record(bad, self.sources)
            self.assertEqual(self.path.read_bytes(), original_file)
        with patch('scene_presence.os.replace', side_effect=OSError('disk')):
            with self.assertRaises(OSError):
                self.ref.record(self.frames, self.sources)
        self.assertEqual(self.path.read_bytes(), original_file)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])
        self.assertEqual(self.ref.assess(self.frames, self.sources).state, 'empty')

    def test_caller_mutation_after_record_does_not_change_reference(self):
        self.ref.record(self.frames, self.sources)
        self.frames['left'][100:160, 100:160] = 0
        self.sources['left'] = 'changed'
        sources = {'left': '/dev/video0', 'right': '/dev/video2'}
        self.assertEqual(self.ref.assess(self.frames, sources).state, 'occupied')
        self.assertEqual(self.ref.assess({'left': background(), 'right': background()}, sources).state, 'empty')

    def test_low_contrast_texture_occlusion_cannot_be_called_empty(self):
        yy, xx = np.indices((480, 640))
        gray = (120 + 16 * ((xx // 20 + yy // 20) % 2)).astype(np.uint8)
        frames = {role: np.repeat(gray[:, :, None], 3, axis=2) for role in ('left', 'right')}
        self.ref.record(frames, self.sources)
        for role in ('left', 'right'):
            changed = {key: value.copy() for key, value in frames.items()}
            changed[role][180:280, 280:380] = 128
            with self.subTest(role=role):
                self.assertNotEqual(self.ref.assess(changed, self.sources).state, 'empty')

    def test_small_global_light_shift_is_unknown(self):
        self.ref.record(self.frames, self.sources)
        for delta in (10, 14, -10, -14):
            frames = {role: (frame.astype(np.int16) + delta).clip(0, 255).astype(np.uint8)
                      for role, frame in self.frames.items()}
            with self.subTest(delta=delta):
                self.assertEqual(self.ref.assess(frames, self.sources).state, 'unknown')

    def test_result_preserves_exact_reference_snapshot_after_rerecord(self):
        self.ref.record(self.frames, self.sources)
        original = self.path.read_bytes()
        result = self.ref.assess(self.frames, self.sources)
        self.assertEqual(result.reference_bytes, original)
        self.assertEqual(result.reference_sha256, sha256(original).hexdigest())
        replacement = {role: np.roll(frame, 10, axis=0) for role, frame in self.frames.items()}
        self.ref.record(replacement, self.sources)
        self.assertNotEqual(self.path.read_bytes(), original)
        self.assertEqual(result.reference_bytes, original)
        self.assertEqual(result.reference_sha256, sha256(original).hexdigest())
        occupied = self.ref.assess(self.frames, self.sources)
        self.assertEqual(occupied.reference_bytes, self.path.read_bytes())
        self.assertEqual(occupied.reference_sha256, sha256(self.path.read_bytes()).hexdigest())

    def test_reference_size_limit_and_missing_reference_have_no_snapshot(self):
        missing = self.ref.assess(self.frames, self.sources)
        self.assertEqual(missing.state, 'unknown')
        self.assertIsNone(missing.reference_bytes)
        self.path.write_bytes(b'x' * (5_000_000 + 1))
        oversized = self.ref.assess(self.frames, self.sources)
        self.assertEqual(oversized.state, 'unknown')
        self.assertIsNone(oversized.reference_bytes)
        self.assertIsNone(oversized.reference_sha256)

    def test_source_mismatch_keeps_snapshot_of_loaded_reference(self):
        self.ref.record(self.frames, self.sources)
        result = self.ref.assess(self.frames, dict(self.sources, left='different'))
        self.assertEqual(result.state, 'unknown')
        self.assertEqual(result.reference_bytes, self.path.read_bytes())
        self.assertEqual(result.reference_sha256, sha256(result.reference_bytes).hexdigest())


if __name__ == '__main__':
    unittest.main()
