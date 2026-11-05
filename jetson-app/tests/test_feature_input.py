import time
import unittest
from types import SimpleNamespace

import app
import numpy as np
from vision_worker import VisionWorker


APP = app.QApplication.instance() or app.QApplication([])


class FeatureRecognizer:
    def __init__(self):
        self.modes = []

    def recognize_pair(self, left, right, preprocessed=False):
        self.modes.append(preprocessed)
        return SimpleNamespace(
            error=None, detection=SimpleNamespace(class_name='yuan_zhu', class_id=0),
            color=None, mask_status=('feature_input', 'feature_input'),
            preview_bgr=np.hstack((left, right)), elapsed_ms=1,
        )

    def close(self):
        pass


class FeatureInputTests(unittest.TestCase):
    def wait(self, predicate):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            APP.processEvents()
            if predicate():
                return True
            time.sleep(0.01)
        return predicate()

    def close_window(self, window):
        window.close()
        self.assertTrue(self.wait(lambda: window._thread is None))

    def test_feature_selection_reaches_worker_and_clears_stale_result(self):
        recognizer = FeatureRecognizer()
        window = app.Window(lambda: recognizer)
        self.addCleanup(self.close_window, window)
        window.display(np.full((320, 640, 3), 77, dtype=np.uint8))
        window.input_mode.setCurrentIndex(1)
        window.recognize()
        self.assertTrue(self.wait(lambda: window.result_label.text() == 'yuan_zhu'))
        self.assertEqual(recognizer.modes, [True])
        self.assertIn('特征', window.color_label.text())
        revision = window.frame_revision
        window.input_mode.setCurrentIndex(0)
        self.assertGreater(window.frame_revision, revision)
        self.assertIsNone(window.preview_view.frame)
        self.assertEqual(window.result_label.text(), '等待识别')

    def test_model_trial_keeps_feature_input_type_and_rejects_rgb_mask_activation(self):
        recognizer = FeatureRecognizer()
        window = app.Window(lambda: recognizer)
        self.addCleanup(self.close_window, window)
        window.display(np.zeros((320, 640, 3), dtype=np.uint8))
        window.input_mode.setCurrentIndex(1)
        dialog = app.ModelDialog(window)
        self.addCleanup(dialog.reject)
        fixture = dialog.fixture()
        self.assertTrue(fixture[2])
        worker = VisionWorker(loader=lambda releases: recognizer)
        _, result = worker._trial({}, fixture)
        self.assertEqual(recognizer.modes, [True])
        self.assertIsNone(result.error)
        with self.assertRaisesRegex(ValueError, 'mask'):
            worker._trial({}, fixture, mask_required=True)


if __name__ == '__main__':
    unittest.main()
