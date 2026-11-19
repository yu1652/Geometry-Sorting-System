"""Desktop conversion/transfer integration without hardware or motion."""
from pathlib import Path
import tempfile
import threading
import unittest

from side_model_manager import QApplication, SideModelDialog
from side_model_store import SideModelStore

APP = QApplication.instance() or QApplication([])


class Receiver:
    def __init__(self):
        self.allowed = 0
        self.jobs = []
        self.closed = False

    def allow(self, seconds=300):
        self.allowed = seconds
        return {'allowed': True, 'seconds_remaining': seconds, 'port': 8788}

    def status(self):
        return {'allowed': bool(self.allowed), 'seconds_remaining': self.allowed, 'port': 8788}

    def update_job(self, job_id, state, message='', details=None):
        self.jobs.append((job_id, state, message, details))

    def close(self):
        self.closed = True


class ConverterUiTests(unittest.TestCase):
    def make_dialog(self, root):
        return SideModelDialog(SideModelStore(root), receiver=Receiver())

    def test_desktop_admission_is_explicit_and_time_limited(self):
        with tempfile.TemporaryDirectory() as directory:
            dialog = self.make_dialog(Path(directory))
            receiver = dialog.receiver
            self.assertEqual(receiver.allowed, 0)
            dialog.receive_button.click()
            self.assertEqual(receiver.allowed, 300)
            self.assertIn('5', dialog.receive_label.text())
            dialog.close()
            self.assertTrue(receiver.closed)

    def test_uploaded_input_mode_is_used_and_activation_remains_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            dialog = self.make_dialog(Path(directory))
            calls = []
            dialog.prepare = lambda path, reuse: calls.append((path, reuse))
            source = Path(directory) / 'new.onnx'
            dialog.on_received(source, 'raw_rgb', 'job1')
            self.assertEqual(dialog.mode.currentIndex(), 1)
            self.assertEqual(calls, [(source, False)])
            self.assertFalse(dialog.confirm_button.isEnabled())
            self.assertFalse(dialog.store.active_path.exists())
            self.assertEqual(dialog.receiver.jobs[-1][1], 'building')
            dialog.close()

    def test_busy_receiver_rejects_new_job_without_starting_another_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            dialog = self.make_dialog(Path(directory))
            # Use a marker, without starting a native GPU worker.
            dialog.worker = object()
            dialog.on_received(Path(directory) / 'second.onnx', 'legacy_features', 'job2')
            self.assertEqual(dialog.receiver.jobs[-1][1], 'failed')
            dialog.worker = None
            dialog.close()

    def test_queued_receive_cannot_start_a_build_after_dialog_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            dialog = self.make_dialog(Path(directory))
            calls = []
            dialog.prepare = lambda path, reuse: calls.append((path, reuse))
            thread = threading.Thread(target=lambda: dialog.bridge.received.emit(
                Path(directory) / 'queued.onnx', 'legacy_features', 'late'))
            thread.start()
            thread.join()
            dialog.close()
            APP.processEvents()
            self.assertEqual(calls, [])
