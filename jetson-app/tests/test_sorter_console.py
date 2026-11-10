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
        editor.setValue(260)
        editor.stepUp()
        self.assertEqual(editor.value(), 260)
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

    def test_import_accepts_machine_offset_beyond_tray_size(self):
        import yaml
        from sorter_console import DEFAULT_POINTS, legacy_rows
        points = [
            {'id': name, 'pixel': [px, py], 'machine': [x + 46, y + 40]}
            for name, px, py, x, y in DEFAULT_POINTS]
        source = self.calibration_path.with_suffix('.yaml')
        source.write_text(yaml.safe_dump({'version': 2, 'unit': 'mm', 'points': points}), encoding='utf8')
        rows = legacy_rows(source)
        self.assertEqual(rows[2][3], 182)
        self.assertEqual(rows[8][4], 176)

    def test_import_windows_v3_points_and_fit_the_profile_detection_roi(self):
        from sorter_console import fit_calibration, legacy_rows

        # Coordinates from the 20261107 Windows calibration export. Its helper
        # tray rectangle differs from the configured automatic detection ROI.
        source = self.calibration_path.with_suffix('.yaml')
        source.write_text('''version: 3
unit: mm
tray_roi: [58, 5, 469, 469]
points:
  - {id: P1, pixel: [126, 77], machine: [85, 26]}
  - {id: P2, pixel: [292, 79], machine: [141, 26]}
  - {id: P3, pixel: [457, 77], machine: [196, 26]}
  - {id: P4, pixel: [130, 244], machine: [85, 81]}
  - {id: P5, pixel: [294, 244], machine: [141, 81]}
  - {id: P6, pixel: [454, 243], machine: [197, 82]}
  - {id: P7, pixel: [132, 406], machine: [85, 136]}
  - {id: P8, pixel: [294, 405], machine: [142, 136]}
  - {id: P9, pixel: [454, 405], machine: [197, 137]}
''', encoding='utf8')
        rows = legacy_rows(source)
        self.assertEqual(rows[0], ('P1', 126, 77, 85, 26))
        with self.assertRaisesRegex(ValueError, 'pixel outside safe region'):
            fit_calibration(rows).map_pixel(126, 77)
        mapping = fit_calibration(rows, roi=(16, 0, 519, 479),
                                  physical_region=(43.563, 0.125, 226.331, 164.027))
        self.assertEqual(tuple(mapping.pixel_region), (16, 0, 535, 479))
        x, y = mapping.map_pixel(126, 77)
        self.assertAlmostEqual(x, 85, delta=0.5)
        self.assertAlmostEqual(y, 26, delta=0.5)
        with self.assertRaisesRegex(ValueError, 'position outside physical region'):
            mapping.validate_physical(227, 80)

        before = self.window.machine.snapshot()
        dialog = self.window.create_calibration_dialog()
        self.addCleanup(self.dispose, dialog)
        dialog.import_points(source)
        self.assertEqual(float(dialog.points.item(0, 1).text()), 126)
        self.assertFalse(self.calibration_path.exists())
        self.assertEqual(self.window.machine.snapshot(), before)

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

    def test_frame_generates_pixel_grid_preserving_machine_coordinates_until_save(self):
        from sorter_console import calibration_rows
        self.window.confirm_origin()
        dialog = self.window.create_calibration_dialog()
        self.addCleanup(self.dispose, dialog)
        dialog.points.item(0, 3).setText('24.123456')
        before = calibration_rows(dialog.points)
        log = self.window.machine.snapshot().log
        dialog.set_frame((180, 50, 300, 300))
        self.assertEqual(calibration_rows(dialog.points), before)
        dialog.generate_points()
        rows = calibration_rows(dialog.points)
        self.assertEqual([(r[1], r[2]) for r in rows],
                         [(x, y) for y in (95, 200, 305) for x in (225, 330, 435)])
        self.assertEqual([r[3:] for r in rows], [r[3:] for r in before])
        self.assertEqual(self.window.machine.snapshot().log, log)
        self.assertFalse(self.calibration_path.exists())
        dialog.points.item(0, 1).setText('226')
        dialog.save_points()
        self.assertTrue(self.calibration_path.exists())
        restored = type(self.window)(calibration_path=self.calibration_path)
        self.addCleanup(self.dispose, restored)
        self.assertEqual(restored.calibration_rows[0][1], 226)
        self.assertEqual(restored.calibration_rows[0][3], 24.123456)
        reloaded = restored.create_calibration_dialog()
        self.addCleanup(self.dispose, reloaded)
        self.assertEqual(calibration_rows(reloaded.points)[0][3], 24.123456)

    def test_frame_grid_validates_bounds_and_inset(self):
        from calibration_frame import frame_grid
        self.assertEqual(frame_grid((0, 0, 639, 479), 0)[-1], (639, 479))
        for box, inset in [((-1, 0, 100, 100), 15), ((10, 0, 640, 100), 15),
                           ((0, 0, 0, 100), 15), ((0, 0, 100, 100), 50),
                           ((0, 0, float('nan'), 100), 15)]:
            with self.subTest(box=box, inset=inset), self.assertRaises(ValueError):
                frame_grid(box, inset)

    def test_saved_frame_and_inset_restore_without_regenerating_adjusted_points(self):
        from sorter_console import calibration_rows
        dialog = self.window.create_calibration_dialog()
        self.addCleanup(self.dispose, dialog)
        dialog.set_frame((150.43, 47.86, 377.44, 374.7))
        dialog.frame_inset.setValue(20)
        dialog.generate_points()
        dialog.points.item(0, 1).setText('227.123456')
        expected_points = calibration_rows(dialog.points)
        dialog.save_points()
        dialog.reject()
        reopened = self.window.create_calibration_dialog()
        self.addCleanup(self.dispose, reopened)
        self.assertEqual(reopened.view.box, (150.43, 47.86, 377.44, 374.7))
        self.assertEqual(reopened.frame_inset.value(), 20)
        self.assertEqual(calibration_rows(reopened.points), expected_points)
        restored = type(self.window)(calibration_path=self.calibration_path)
        self.addCleanup(self.dispose, restored)
        restarted = restored.create_calibration_dialog()
        self.addCleanup(self.dispose, restarted)
        self.assertEqual(restarted.view.box, reopened.view.box)
        self.assertEqual(restarted.frame_inset.value(), 20)
        self.assertEqual(calibration_rows(restarted.points), expected_points)
        # Unsaved edits do not overwrite the last saved settings.
        restarted.set_frame((160, 60, 300, 300))
        restarted.frame_inset.setValue(10)
        restarted.reject()
        discarded = restored.create_calibration_dialog()
        self.addCleanup(self.dispose, discarded)
        self.assertEqual(discarded.view.box, reopened.view.box)
        self.assertEqual(discarded.frame_inset.value(), 20)

    def test_old_point_file_without_frame_keeps_coordinates_and_uses_default_box(self):
        import json
        from sorter_console import DEFAULT_POINTS, TRAY_ROI
        rows = [list(row) for row in DEFAULT_POINTS]
        rows[0][1] = 212
        self.calibration_path.write_text(json.dumps({'mode':'simulation', 'points':rows}), encoding='utf8')
        restored = type(self.window)(calibration_path=self.calibration_path)
        self.addCleanup(self.dispose, restored)
        dialog = restored.create_calibration_dialog()
        self.addCleanup(self.dispose, dialog)
        self.assertEqual(dialog.view.box, TRAY_ROI)
        self.assertEqual(float(dialog.points.item(0, 1).text()), 212)

    def test_frame_drag_maps_letterboxed_preview_and_does_not_move_machine(self):
        try:
            from PySide6.QtCore import QPoint, Qt
            from PySide6.QtTest import QTest
        except ImportError:
            from PyQt5.QtCore import QPoint, Qt
            from PyQt5.QtTest import QTest
        dialog = self.window.create_calibration_dialog()
        self.addCleanup(self.dispose, dialog)
        view = dialog.view
        # Exercise a wide preview with horizontal letterboxing.
        view.setParent(None)
        self.addCleanup(self.dispose, view)
        view.resize(800, 480)
        view.show()
        self.app.processEvents()
        self.assertIsNone(view.image_position(20, 20))
        self.assertEqual(view.image_position(280, 100), (200, 100))
        before = self.window.machine.snapshot().log
        rows = [dialog.points.item(0, c).text() for c in range(5)]
        left = getattr(Qt, 'MouseButton', Qt).LeftButton
        # Draw a new box outside the initial box, then translate its interior.
        QTest.mousePress(view, left, pos=QPoint(90, 10))
        QTest.mouseMove(view, QPoint(210, 130))
        QTest.mouseRelease(view, left, pos=QPoint(210, 130))
        self.assertEqual(view.box, (10, 10, 120, 120))
        QTest.mousePress(view, left, pos=QPoint(150, 70))
        QTest.mouseMove(view, QPoint(180, 90))
        QTest.mouseRelease(view, left, pos=QPoint(180, 90))
        self.assertEqual(view.box, (40, 30, 120, 120))
        # Resize the bottom right corner.
        QTest.mousePress(view, left, pos=QPoint(240, 150))
        QTest.mouseMove(view, QPoint(260, 170))
        QTest.mouseRelease(view, left, pos=QPoint(260, 170))
        self.assertEqual(view.box, (40, 30, 140, 140))
        self.assertEqual(self.window.machine.snapshot().log, before)
        self.assertEqual([dialog.points.item(0, c).text() for c in range(5)], rows)
        self.assertEqual(dialog.frame_fields[0].value(), 40)
        self.assertEqual(dialog.frame_fields[2].value(), 140)


if __name__ == "__main__":
    unittest.main()
