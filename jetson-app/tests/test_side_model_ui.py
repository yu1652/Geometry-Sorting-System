from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from side_model_manager import QApplication, SideModelDialog
from side_model_store import SideModelStore
from test_live_side_model import Detector, make_bundle

APP = QApplication.instance() or QApplication([])


class ModelUiTests(unittest.TestCase):
    def wait(self, condition):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            APP.processEvents()
            if condition():
                return
            time.sleep(.005)
        self.fail('background model operation did not finish')

    def test_trial_runs_off_ui_thread_and_activation_requires_confirm(self):
        class Probe(Detector):
            def infer(self, tensor):
                self.infer_thread = threading.get_ident()
                return super().infer(tensor)
            def close(self):
                self.close_thread = threading.get_ident()
                super().close()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            make_bundle(root / 'offline')
            store = SideModelStore(root / 'live')
            dialog = SideModelDialog(store)
            self.addCleanup(dialog.close)
            probe = Probe()
            with patch('side_model_store.create_backend', return_value=probe):
                dialog.prepare(root / 'offline', True)
                self.wait(lambda: dialog.worker is None)
            self.assertFalse(store.active_path.exists())
            self.assertTrue(dialog.confirm_button.isEnabled())
            self.assertFalse(dialog.mode.isEnabled())
            self.assertNotEqual(probe.infer_thread, threading.get_ident())
            self.assertEqual(probe.infer_thread, probe.close_thread)
            dialog.confirm_button.click()
            self.assertEqual(len(store.read_active()['class_names']), 10)
            self.assertIn('重启', dialog.status.text())

    def test_failed_import_unblocks_controls_without_enabling_candidate(self):
        with tempfile.TemporaryDirectory() as d:
            store = SideModelStore(Path(d) / 'live')
            dialog = SideModelDialog(store)
            self.addCleanup(dialog.close)
            dialog.prepare(Path(d) / 'missing', True)
            self.wait(lambda: dialog.worker is None)
            self.assertFalse(dialog.confirm_button.isEnabled())
            self.assertTrue(dialog.reuse_button.isEnabled())
            self.assertFalse(store.active_path.exists())
