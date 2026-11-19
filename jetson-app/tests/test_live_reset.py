import unittest

from live_control import LiveController


class KnownOriginResetTests(unittest.TestCase):
    def setUp(self):
        self.sent = []
        self.now = 0.0
        self.control = LiveController(self.sent.append, clock=lambda: self.now)
        self.control.begin_handshake()
        for line in ('PONG', 'VERSION,ESP32_SORTER_XY_PUSHER,0.1.0',
                     'STATUS,MCU,WAIT_START,0,0,77'):
            self.control.handle_mcu(line)
        self.control.confirm_origin()
        self.control.handle_mcu('READY')
        self.control.goto(40, 30)
        self.control.handle_mcu('STATUS,MCU,WAIT_START,1600,1170,77')
        self.control._counts = [2, 1, 0, 0, 0, 0]
        self.sent.clear()

    def reply(self, *lines):
        for line in lines:
            self.control.handle_mcu(line)

    def stopped(self, x=1600, y=1170):
        self.reply('RX,STOP', 'STATUS,PUSHER_UP', 'STATUS,STOP',
                   'RX,STATUS?', 'STATUS,MCU,STOP,%d,%d,77' % (x, y))

    def lifted(self):
        self.reply('RX,PUSHER,UP', 'STATUS,PUSHER_UP')

    def home_arrived(self):
        self.reply('RX,GOTO_MM,0.000,0.000', 'RX,STATUS?', 'STATUS,MCU,STOP,0,0,77')

    def reset_verified(self):
        self.reply('RX,RESET', 'READY', 'STATUS,WAIT_START',
                   'RX,STATUS?', 'STATUS,MCU,WAIT_START,0,0,77')

    def test_normal_stop_keeps_known_boot_and_full_reset_returns_home_before_reset(self):
        self.assertTrue(self.control.can_reuse_origin)
        self.control.stop()
        self.reply('STATUS,STOP', 'STATUS,MCU,STOP,1600,1170,77')
        self.assertTrue(self.control.can_reuse_origin)
        self.assertFalse(self.control.snapshot().origin_confirmed)
        self.assertTrue(self.control.reset_known_origin())
        self.assertTrue(self.control.reset_in_progress)
        self.assertEqual(self.sent[-1], 'STOP')
        self.stopped()
        self.assertEqual(self.sent[-1], 'PUSHER,UP')
        self.lifted()
        self.assertEqual(self.sent[-2:], ['GOTO_MM,0.000,0.000', 'STATUS?'])
        self.assertNotIn('RESET', self.sent)
        self.home_arrived()
        self.assertEqual(self.sent[-1], 'RESET')
        self.assertEqual(self.control.snapshot().completed_count, 3)
        self.reset_verified()
        self.assertFalse(self.control.reset_in_progress)
        self.assertTrue(self.control.snapshot().origin_confirmed)
        self.assertEqual(self.control.snapshot().state, 'armed')
        self.assertEqual(self.control.snapshot().completed_count, 0)
        self.assertEqual(self.control.reset_generation, 1)
        self.assertNotIn('START', self.sent)

    def test_at_home_skips_motion_and_can_preserve_counts(self):
        self.assertTrue(self.control.reset_known_origin(clear_counts=False))
        self.stopped(0, 0)
        self.lifted()
        self.assertEqual(self.sent[-1], 'RESET')
        self.assertFalse(any(line.startswith('GOTO') for line in self.sent))
        self.reset_verified()
        self.assertEqual(self.control.snapshot().completed_count, 3)
        self.assertEqual(self.control.reset_generation, 1)

    def test_stale_stop_up_zero_and_ready_cannot_advance_reset(self):
        self.control.reset_known_origin()
        before = list(self.sent)
        self.reply('STATUS,STOP', 'STATUS,PUSHER_UP', 'READY', 'STATUS,MCU,STOP,0,0,77')
        self.assertEqual(self.sent, before)
        self.stopped()
        before = list(self.sent)
        self.reply('STATUS,PUSHER_UP', 'STATUS,MCU,STOP,0,0,77')
        self.assertEqual(self.sent, before)
        self.lifted()
        before = list(self.sent)
        self.reply('STATUS,MCU,STOP,0,0,77', 'RX,GOTO_MM,0.000,0.000',
                   'STATUS,MCU,STOP,0,0,77', 'RX,STATUS?', 'STATUS,MCU,STOP,0,1,77')
        self.assertEqual(self.sent, before)
        self.reply('STATUS,MCU,STOP,0,0,77')
        self.assertEqual(self.sent[-1], 'RESET')
        self.reply('READY', 'STATUS,MCU,WAIT_START,0,0,77')
        self.assertEqual(self.control.reset_generation, 0)
        self.assertFalse(self.control.snapshot().origin_confirmed)
        self.reply('RX,RESET', 'READY', 'STATUS,MCU,WAIT_START,0,0,77')
        self.assertEqual(self.control.reset_generation, 0)
        self.reply('RX,STATUS?', 'STATUS,MCU,WAIT_START,0,0,77')
        self.assertEqual(self.control.reset_generation, 1)

    def test_boot_change_invalidates_anchor_and_never_rebases_unknown_position(self):
        self.control.reset_known_origin()
        self.stopped()
        self.lifted()
        self.reply('STATUS,MCU,STOP,0,0,78')
        self.assertFalse(self.control.reset_in_progress)
        self.assertFalse(self.control.can_reuse_origin)
        self.assertEqual(self.control.reset_generation, 0)
        self.assertEqual(self.control.snapshot().completed_count, 3)
        self.assertNotIn('RESET', self.sent)

    def test_disconnect_fault_timeout_and_explicit_stop_cancel_without_clearing_counts(self):
        for cause in ('disconnect', 'fault', 'timeout', 'stop'):
            with self.subTest(cause=cause):
                self.setUp()
                self.control.reset_known_origin()
                self.stopped()
                if cause == 'disconnect':
                    self.control.disconnect()
                elif cause == 'fault':
                    self.reply('ERROR,TEST')
                elif cause == 'timeout':
                    self.now = 31
                    self.control.tick()
                else:
                    self.control.stop()
                self.assertFalse(self.control.reset_in_progress)
                self.assertFalse(self.control.can_reuse_origin)
                self.reply('RX,PUSHER,UP', 'STATUS,PUSHER_UP', 'RX,RESET', 'READY')
                self.assertEqual(self.control.snapshot().completed_count, 3)
                self.assertEqual(self.control.reset_generation, 0)
                self.assertFalse(self.control.snapshot().origin_confirmed)

    def test_reset_rejects_missing_anchor_and_pending_manual_motion(self):
        self.control.goto(20, 20)
        before = list(self.sent)
        self.assertFalse(self.control.reset_known_origin())
        self.assertEqual(self.sent, before)
        self.control.disconnect()
        self.assertFalse(self.control.reset_known_origin())
        self.control.begin_handshake()
        self.reply('PONG', 'VERSION,ESP32_SORTER_XY_PUSHER,0.1.0',
                   'STATUS,MCU,WAIT_START,800,780,77')
        before = list(self.sent)
        self.assertFalse(self.control.reset_known_origin())
        self.assertEqual(self.sent, before)

    def test_reported_origin_change_stops_and_invalidates_anchor_without_rebasing_or_clearing(self):
        before_epoch = self.control.epoch
        self.assertTrue(self.control.invalidate_origin())
        self.assertEqual(self.sent, ['STOP', 'STATUS?'])
        self.assertGreater(self.control.epoch, before_epoch)
        self.assertFalse(self.control.snapshot().origin_confirmed)
        self.assertFalse(self.control.can_reuse_origin)
        self.assertFalse(self.control.reset_in_progress)
        self.assertEqual(self.control.snapshot().last_error, '零点或机构位置已改变')
        self.assertEqual(self.control.snapshot().completed_count, 3)
        self.assertEqual((self.control.snapshot().x_mm, self.control.snapshot().y_mm), (40, 30))
        self.reply('STATUS,STOP', 'STATUS,MCU,STOP,1600,1170,77')
        before = list(self.sent)
        self.assertFalse(self.control.reset_known_origin())
        self.assertFalse(self.control.start())
        self.assertEqual(self.sent, before)
        self.assertEqual(self.control.reset_generation, 0)


if __name__ == '__main__':
    unittest.main()
