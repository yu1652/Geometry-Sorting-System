import os
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from app import QApplication, qt_enum
from live_console import LiveWindow
from sorter_console import QSizePolicy


class JP4CalibrationTests(unittest.TestCase):
    def test_calibration_opens_with_qt59_empty_policy_namespace(self):
        application = QApplication.instance() or QApplication([])
        ignored = qt_enum(QSizePolicy, 'Policy', 'Ignored')

        class OldSizePolicy:
            class Policy:
                pass
            Ignored = ignored

        with tempfile.TemporaryDirectory() as folder:
            window = LiveWindow(data_dir=folder, butu_24=True)
            try:
                with patch('sorter_console.QSizePolicy', OldSizePolicy):
                    dialog = window.create_calibration_dialog()
                try:
                    self.assertEqual(dialog.points.rowCount(), 9)
                    self.assertEqual(dialog.view.sizePolicy().horizontalPolicy(), ignored)
                finally:
                    dialog.close()
            finally:
                window.close()


if __name__ == '__main__':
    unittest.main()
