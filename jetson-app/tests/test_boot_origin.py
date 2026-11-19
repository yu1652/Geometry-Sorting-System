from itertools import permutations
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from app import QApplication, QMessageBox
from esp32_serial import SerialLineTransport
from live_console import LiveWindow, _claim_fixed_origin_boot
from live_control import LiveController


APP = QApplication.instance() or QApplication([])
VERSION = 'VERSION,ESP32_SORTER_XY_PUSHER,0.1.0'


class FixedOriginBootClaimTests(unittest.TestCase):
    def test_one_claim_per_board_boot(self):
        with tempfile.TemporaryDirectory() as runtime, \
                patch('live_console.tempfile.gettempdir', return_value=runtime), \
                patch('live_console.os.getuid', return_value=1000, create=True), \
                patch('live_console.Path.read_text', side_effect=('boot-a\n', 'boot-a\n', 'boot-b\n')):
            self.assertTrue(_claim_fixed_origin_boot())
            self.assertFalse(_claim_fixed_origin_boot())
            self.assertTrue(_claim_fixed_origin_boot())


class BootOriginTests(unittest.TestCase):
    def setUp(self):
        self.commands = []
        self.control = LiveController(self.commands.append, accept_boot_origin=True)

    def handshake(self, status='STATUS,MCU,WAIT_START,0,0,77', order=None):
        self.control.begin_handshake()
        for reply in order or ('PONG', VERSION, status):
            self.control.handle_mcu(reply)

    def test_fresh_esp32_zero_is_accepted_without_reset_or_motion(self):
        self.handshake()
        snapshot = self.control.snapshot()
        self.assertTrue(snapshot.origin_confirmed)
        self.assertEqual(snapshot.state, 'armed')
        self.assertEqual((snapshot.x_mm, snapshot.y_mm), (0.0, 0.0))
        self.assertTrue(self.control.can_reuse_origin)
        self.assertEqual(self.commands, ['PING', 'VERSION?', 'STATUS?'])

    def test_all_handshake_reply_orders_accept_only_after_complete(self):
        replies = ('PONG', VERSION, 'STATUS,MCU,WAIT_START,0,0,77')
        for order in permutations(replies):
            with self.subTest(order=order):
                commands = []
                control = LiveController(commands.append, accept_boot_origin=True)
                control.begin_handshake()
                for reply in order[:-1]:
                    control.handle_mcu(reply)
                    self.assertFalse(control.snapshot().origin_confirmed)
                control.handle_mcu(order[-1])
                self.assertTrue(control.snapshot().origin_confirmed)
                self.assertEqual(commands, ['PING', 'VERSION?', 'STATUS?'])

    def test_nonzero_startup_report_is_not_rebased(self):
        self.handshake('STATUS,MCU,WAIT_START,1600,1170,77')
        snapshot = self.control.snapshot()
        self.assertTrue(self.control.ready)
        self.assertFalse(snapshot.origin_confirmed)
        self.assertEqual((snapshot.x_mm, snapshot.y_mm), (40.0, 30.0))
        self.assertNotIn('RESET', self.commands)

    def test_stopped_or_running_zero_is_not_treated_as_power_on(self):
        for state in ('STOP', 'PUSHING', 'WAIT_SORT', 'SORTING'):
            with self.subTest(state=state):
                commands = []
                control = LiveController(commands.append, accept_boot_origin=True)
                control.begin_handshake()
                for reply in ('PONG', VERSION, 'STATUS,MCU,%s,0,0,77' % state):
                    control.handle_mcu(reply)
                self.assertFalse(control.snapshot().origin_confirmed)
                self.assertNotIn('RESET', commands)

    def test_new_esp32_boot_accepts_its_new_power_on_zero(self):
        self.handshake()
        self.control.handle_mcu('BOOT,88')
        self.assertFalse(self.control.snapshot().origin_confirmed)
        for reply in ('PONG', VERSION, 'STATUS,MCU,WAIT_START,0,0,88'):
            self.control.handle_mcu(reply)
        self.assertTrue(self.control.snapshot().origin_confirmed)
        self.assertTrue(self.control.can_reuse_origin)
        self.assertNotIn('RESET', self.commands)

    def test_disconnect_and_same_boot_do_not_silently_reconfirm_origin(self):
        self.handshake()
        self.control.disconnect()
        self.handshake()
        self.assertFalse(self.control.snapshot().origin_confirmed)
        self.assertFalse(self.control.can_reuse_origin)
        self.assertNotIn('RESET', self.commands)

    def test_explicit_origin_invalidation_is_not_undone_by_same_boot(self):
        self.handshake()
        self.control.invalidate_origin()
        self.handshake()
        self.assertFalse(self.control.snapshot().origin_confirmed)
        self.assertFalse(self.control.can_reuse_origin)

    def test_later_zero_does_not_replace_nonzero_first_observation(self):
        self.handshake('STATUS,MCU,WAIT_START,1600,1170,77')
        self.handshake()
        self.assertFalse(self.control.snapshot().origin_confirmed)
        self.assertNotIn('RESET', self.commands)

    def test_incompatible_firmware_cannot_authorize_boot_origin(self):
        self.control.begin_handshake()
        for reply in ('PONG', 'VERSION,ESP32_SORTER_XY_PUSHER,0.1.1',
                      'STATUS,MCU,WAIT_START,0,0,77'):
            self.control.handle_mcu(reply)
        self.assertFalse(self.control.ready)
        self.assertFalse(self.control.snapshot().origin_confirmed)

    def test_missing_boot_identifier_cannot_authorize_origin(self):
        self.control.begin_handshake()
        for reply in ('PONG', VERSION, 'STATUS,MCU,WAIT_START,0,0'):
            self.control.handle_mcu(reply)
        self.assertFalse(self.control.ready)
        self.assertFalse(self.control.snapshot().origin_confirmed)

    def test_boot_origin_does_not_start_sorting_automatically(self):
        self.handshake()
        self.assertTrue(self.control.start())
        self.assertEqual(self.commands, ['PING', 'VERSION?', 'STATUS?', 'HOST_READY'])
        self.assertNotIn('START', self.commands)

    def test_manual_origin_controller_mode_is_unchanged(self):
        commands = []
        control = LiveController(commands.append)
        control.begin_handshake()
        for reply in ('PONG', VERSION, 'STATUS,MCU,WAIT_START,0,0,77'):
            control.handle_mcu(reply)
        self.assertFalse(control.snapshot().origin_confirmed)
        self.assertTrue(control.confirm_origin())
        self.assertEqual(commands[-1], 'RESET')


class Port:
    def __init__(self):
        self.commands = []
        self.data = b''

    @property
    def in_waiting(self):
        return len(self.data)

    def read(self, size):
        result, self.data = self.data[:size], self.data[size:]
        return result

    def write(self, data):
        self.commands.extend(data.decode('ascii').splitlines())
        return len(data)

    def close(self):
        pass


class Vision:
    def shutdown(self):
        pass


class BootOriginUiTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.window = LiveWindow(data_dir=Path(self.folder.name), butu_24=True)
        self.window.timer.stop()
        self.port = Port()
        self.window.serial_port = self.port
        self.window.transport = SerialLineTransport(self.port)

    def tearDown(self):
        self.window.vision = None
        self.window._finish_serial_close()
        self.window.close()
        self.folder.cleanup()

    def handshake(self, status):
        self.window.machine.begin_handshake()
        for reply in ('PONG', VERSION, status):
            self.window.machine.handle_mcu(reply)

    def test_real_console_accepts_boot_zero_and_start_needs_no_manual_dialog(self):
        self.handshake('STATUS,MCU,WAIT_START,0,0,77')
        self.assertTrue(self.window.machine.snapshot().origin_confirmed)
        self.window.save_calibration(self.window.calibration_rows)
        self.window.vision = Vision()
        self.window._frames_at = time.monotonic()
        self.window._models_warm = True
        cancel = getattr(QMessageBox, 'Cancel', None)
        if cancel is None:
            cancel = QMessageBox.StandardButton.Cancel
        with patch('live_console.QMessageBox.question', return_value=cancel):
            self.assertTrue(self.window.start())
        self.assertEqual(self.port.commands, ['PING', 'VERSION?', 'STATUS?', 'HOST_READY'])

    def test_reopening_console_at_nonzero_position_does_not_zero_mcu_counters(self):
        self.handshake('STATUS,MCU,WAIT_START,1600,1170,77')
        snapshot = self.window.machine.snapshot()
        self.assertFalse(snapshot.origin_confirmed)
        self.assertEqual((snapshot.x_mm, snapshot.y_mm), (40.0, 30.0))
        self.assertEqual(self.port.commands, ['PING', 'VERSION?', 'STATUS?'])

    def receive(self, *lines):
        self.port.data = ('\n'.join(lines) + '\n').encode('ascii')
        self.window.poll_serial()

    def test_fixed_origin_boot_can_confirm_stopped_zero_once(self):
        self.window.save_calibration(self.window.calibration_rows)
        self.window._startup_zero_reset_pending = True
        self.window.machine.begin_handshake()
        with patch('live_console.QMessageBox.question') as question:
            self.receive('PONG', VERSION, 'STATUS,MCU,STOP,0,0,77')
            self.assertFalse(self.window.machine.snapshot().origin_confirmed)
            self.assertEqual(self.port.commands, ['PING', 'VERSION?', 'STATUS?', 'RESET'])
            self.receive('STATUS,MCU,STOP,0,0,77', 'READY')
        question.assert_not_called()
        self.assertTrue(self.window.machine.snapshot().origin_confirmed)
        self.assertEqual(self.port.commands, ['PING', 'VERSION?', 'STATUS?', 'RESET'])

    def test_fixed_origin_boot_does_not_relabel_nonzero_or_later_status(self):
        self.window.save_calibration(self.window.calibration_rows)
        self.window._startup_zero_reset_pending = True
        self.window.machine.begin_handshake()
        self.receive('PONG', VERSION, 'STATUS,MCU,STOP,1960,195,77')
        self.receive('STATUS,MCU,STOP,0,0,77')
        self.assertFalse(self.window.machine.snapshot().origin_confirmed)
        self.assertEqual(self.port.commands, ['PING', 'VERSION?', 'STATUS?'])


if __name__ == '__main__':
    unittest.main()
