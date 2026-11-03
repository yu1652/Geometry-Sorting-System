import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import ReplayState, save_sample


class CoreTests(unittest.TestCase):
    def test_empty_replay_cannot_play(self):
        with self.assertRaises(ValueError):
            ReplayState().play()

    def test_pause_stop_preserve_loaded_source(self):
        s = ReplayState()
        s.load('photo.jpg')
        s.play()
        self.assertEqual(s.status, 'playing')
        s.pause()
        self.assertEqual(s.status, 'paused')
        s.stop()
        self.assertEqual(s.status, 'stopped')
        self.assertEqual(s.source, 'photo.jpg')

    def test_save_keeps_source_and_never_overwrites_previous(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = save_sample(Path(tmp), '原图.jpg', b'image-bytes')
            b = save_sample(Path(tmp), 'second.jpg', b'next')
            self.assertNotEqual(a, b)
            self.assertEqual((a/'source.png').read_bytes(), b'image-bytes')
            meta = json.loads((a/'metadata.json').read_text(encoding='utf8'))
            self.assertEqual(meta['source'], '原图.jpg')
            self.assertFalse(meta['hardware_enabled'])

    def test_empty_sample_is_rejected_without_creating_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                save_sample(Path(tmp), 'a.jpg', b'')
            self.assertEqual(list(Path(tmp).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
