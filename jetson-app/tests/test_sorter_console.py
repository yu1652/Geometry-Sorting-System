import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from app import QApplication
try:
    from PySide6.QtCore import QCoreApplication, QEvent
except ImportError:
    from PyQt5.QtCore import QCoreApplication, QEvent


class SimulationConsoleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from sorter_console import SimulationWindow

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.calibration_path = Path(self.temp.name) / "calibration.json"
        self.window = SimulationWindow(calibration_path=self.calibration_path)
        self.addCleanup(self.dispose, self.window)

    def dispose(self, widget):
        widget.close()
        widget.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    def test_requires_manual_origin_and_updates_competition_only_after_cycle(self):
        self.window.run_cycle()
        self.assertEqual(self.window.records.rowCount(), 0)
        self.window.confirm_origin()
        self.window.start()
        self.window.run_cycle()
        self.assertEqual(self.window.records.rowCount(), 1)
        self.assertEqual(self.window.total_label.text(), "成功总数 1")
        self.window.show_competition()
        self.assertIs(self.window.pages.currentWidget(), self.window.competition_page)

    def test_nine_point_edit_requires_save_and_reloads(self):
        dialog = self.window.create_calibration_dialog()
        self.addCleanup(self.dispose, dialog)
        self.assertEqual(dialog.points.rowCount(), 9)
        self.assertFalse(self.calibration_path.exists())
        dialog.points.item(0, 1).setText("211.25")
        dialog.save_points()
        self.assertTrue(self.calibration_path.exists())
        restored = type(self.window)(calibration_path=self.calibration_path)
        self.addCleanup(self.dispose, restored)
        saved = restored.create_calibration_dialog()
        self.addCleanup(self.dispose, saved)
        self.assertEqual(float(saved.points.item(0, 1).text()), 211.25)

    def test_calibration_hold_closes_with_pusher_raised(self):
        self.window.confirm_origin()
        dialog = self.window.create_calibration_dialog()
        self.addCleanup(self.dispose, dialog)
        dialog.move_pusher(True)
        self.assertTrue(self.window.machine.snapshot().pusher_down)
        dialog.reject()
        self.assertFalse(self.window.machine.snapshot().pusher_down)

    def test_legacy_import_is_draft_until_saved(self):
        import yaml
        from sorter_console import DEFAULT_POINTS

        points = [
            {"id": name, "pixel": [px, py], "machine": [x, y]}
            for name, px, py, x, y in DEFAULT_POINTS
        ]
        points[0]["pixel"][0] = 212
        source = self.calibration_path.with_suffix(".yaml")
        source.write_text(yaml.safe_dump({"version": 2, "unit": "mm", "points": points}), encoding="utf-8")
        dialog = self.window.create_calibration_dialog()
        self.addCleanup(self.dispose, dialog)
        dialog.import_points(source)
        self.assertEqual(float(dialog.points.item(0, 1).text()), 212)
        self.assertFalse(self.calibration_path.exists())
        dialog.save_points()
        self.assertTrue(self.calibration_path.exists())


if __name__ == "__main__":
    unittest.main()
