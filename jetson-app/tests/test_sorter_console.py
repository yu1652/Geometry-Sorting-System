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

    def test_numeric_arrows_update_draft_without_moving_and_preserve_on_save(self):
        from sorter_console import CoordinateSpinBox, calibration_rows
        self.window.confirm_origin()
        dialog = self.window.create_calibration_dialog()
        self.addCleanup(self.dispose, dialog)
        pixel = dialog.findChild(CoordinateSpinBox, 'coordinate_0_1')
        mechanical = dialog.findChild(CoordinateSpinBox, 'coordinate_0_3')
        before = self.window.machine.snapshot()
        self.assertEqual(pixel.singleStep(), 1)
        self.assertEqual(mechanical.singleStep(), 0.1)
        pixel.stepUp()
        mechanical.stepUp()
        self.assertEqual(calibration_rows(dialog.points)[0][1], 212)
        self.assertAlmostEqual(calibration_rows(dialog.points)[0][3], 24.1)
        self.assertEqual(self.window.machine.snapshot().log, before.log)
        dialog.save_points()
        restored = type(self.window)(calibration_path=self.calibration_path)
        self.addCleanup(self.dispose, restored)
        saved = restored.create_calibration_dialog()
        self.addCleanup(self.dispose, saved)
        self.assertAlmostEqual(saved.findChild(CoordinateSpinBox, 'coordinate_0_3').value(), 24.1)

    def test_wheel_selects_point_updates_value_and_enforces_limits(self):
        from sorter_console import CoordinateSpinBox, calibration_rows
        try:
            from PySide6.QtCore import QPoint, QPointF, Qt
            from PySide6.QtGui import QWheelEvent
        except ImportError:
            from PyQt5.QtCore import QPoint, QPointF, Qt
            from PyQt5.QtGui import QWheelEvent
        dialog = self.window.create_calibration_dialog()
        self.addCleanup(self.dispose, dialog)
        dialog.show()
        self.app.processEvents()
        editor = dialog.findChild(CoordinateSpinBox, 'coordinate_4_3')
        event = QWheelEvent(QPointF(10, 10), QPointF(10, 10), QPoint(), QPoint(0, 120),
                            getattr(Qt, 'MouseButton', Qt).NoButton,
                            getattr(Qt, 'KeyboardModifier', Qt).NoModifier,
                            getattr(Qt, 'ScrollPhase', Qt).NoScrollPhase, False)
        self.app.sendEvent(editor, event)
        self.assertEqual(dialog.points.currentRow(), 4)
        self.assertAlmostEqual(calibration_rows(dialog.points)[4][3], 80.1)
        editor.setValue(160)
        editor.stepUp()
        self.assertEqual(editor.value(), 160)
        editor.setValue(0)
        editor.stepDown()
        self.assertEqual(editor.value(), 0)

    def test_typed_value_is_used_by_explicit_goto(self):
        from sorter_console import CoordinateSpinBox
        self.window.confirm_origin()
        dialog = self.window.create_calibration_dialog()
        self.addCleanup(self.dispose, dialog)
        editor = dialog.findChild(CoordinateSpinBox, 'coordinate_0_3')
        editor.lineEdit().setText('32.7')
        editor.interpretText()
        self.assertEqual(self.window.machine.snapshot().x_mm, 0)
        dialog.goto_selected()
        self.assertAlmostEqual(self.window.machine.snapshot().x_mm, 32.7)

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
