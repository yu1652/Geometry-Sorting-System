import os
import tempfile
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from app import QApplication, QPushButton, Qt
from live_console import LiveWindow
from sorter_console import calibration_rows


class ReleaseCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.window = LiveWindow(data_dir=folder.name, linux_20261107=True)
        self.addCleanup(self.window.close)
        self.machine = self.window.machine
        self.sent = []
        self.machine._send = self.sent.append
        self.dialog = self.window.create_calibration_dialog()
        self.addCleanup(self.dialog.reject)

    def click(self, text):
        control = next(button for button in self.dialog.findChildren(QPushButton) if button.text() == text)
        control.click()

    def test_test_move_and_return_to_zero_share_live_goto_gates(self):
        self.dialog.test_x.setValue(182)
        self.dialog.test_y.setValue(12)
        self.click('GOTO 测试坐标')
        self.click('回 HOME')
        self.assertEqual(self.sent, [])

        self.machine.begin_handshake()
        self.addCleanup(self.machine.disconnect)
        for line in ('PONG', 'VERSION,ESP32_SORTER_XY_PUSHER,0.1.0', 'STATUS,MCU,WAIT_START,0,0,77'):
            self.machine.handle_mcu(line)
        self.machine.confirm_origin()
        self.machine.handle_mcu('READY')
        self.machine.set_pusher(True)
        self.machine.handle_mcu('STATUS,PUSHER_DOWN')
        before = list(self.sent)
        self.click('回 HOME')
        self.assertEqual(self.sent, before)
        self.machine.set_pusher(False)
        self.machine.handle_mcu('STATUS,PUSHER_UP')

        try:
            from PySide6.QtTest import QTest
        except ImportError:
            from PyQt5.QtTest import QTest
        self.dialog.show()
        self.dialog.test_x.lineEdit().setFocus()
        before = list(self.sent)
        QTest.keyClick(self.dialog.test_x.lineEdit(), getattr(Qt, 'Key', Qt).Key_Return)
        self.assertEqual(self.sent, before)

        self.click('GOTO 测试坐标')
        self.assertEqual(self.sent[-2:], ['GOTO_MM,182.000,12.000', 'STATUS?'])
        self.assertEqual(self.machine.pending, 'goto')
        self.machine.handle_mcu('STATUS,MCU,WAIT_START,7280,468,77')
        self.click('回 HOME')
        self.assertEqual(self.sent[-2:], ['GOTO_MM,0.000,0.000', 'STATUS?'])
        self.assertEqual(self.machine.pending, 'goto')

    def test_write_back_next_and_atomic_translation_only_change_the_draft(self):
        points = self.dialog.points
        points.setCurrentCell(1, 3)
        self.dialog.test_x.setValue(259)
        self.dialog.test_y.setValue(229)
        self.click('测试 X/Y 写回当前点')
        before = calibration_rows(points)
        self.assertEqual(before[1][3:], (259, 229))
        self.click('下一点')
        self.assertEqual(points.currentRow(), 2)
        self.dialog.shift_step.setValue(2)
        for direction in ('X+', 'Y+'):
            self.click(direction)
            self.assertEqual(calibration_rows(points), before)

        self.dialog.shift_step.setValue(1)
        self.click('X+')
        self.click('Y-')
        self.assertEqual(calibration_rows(points), [(name, px, py, x + 1, y - 1)
                                                   for name, px, py, x, y in before])
        points.item(2, 1).setText('476.5')
        self.click('撤销平移')
        restored = calibration_rows(points)
        self.assertEqual([row[3:] for row in restored], [row[3:] for row in before])
        self.assertEqual(restored[2][1], 476.5)
        self.assertEqual(self.sent, [])
        self.assertFalse(self.window.calibration_path.exists())
        self.click('保存标定')
        self.assertEqual(self.window.calibration_rows, restored)
        self.assertTrue(self.window.calibration_path.exists())
        self.assertEqual(self.sent, [])
