"""Unknown-part gate disposal uses real protocol replies, never SORT or DONE."""
import unittest

from live_control import LiveController


class UnknownScanRejectTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.sent = []
        self.control = LiveController(self.sent.append, clock=lambda: self.now,
                                      timeout=30, scan_timeout=5, done_timeout=5)

    def reach_scan(self):
        control = self.control
        control.begin_handshake()
        for line in ('PONG', 'VERSION,ESP32_SORTER_XY_PUSHER,0.1.0',
                     'STATUS,MCU,WAIT_START,0,0,123'):
            control.handle_mcu(line)
        self.assertTrue(control.confirm_origin())
        control.handle_mcu('READY')
        self.assertTrue(control.start())
        control.handle_mcu('STATUS,HOST_READY')
        control.handle_mcu('STATUS,START')
        control.handle_mcu('PUSH?')
        self.assertTrue(control.provide_target(120, 60))
        control.handle_mcu('SCAN')
        control.handle_mcu('STATUS,MCU,WAIT_SORT,1960,195,123')
        self.sent.clear()

    def begin_reject(self):
        self.reach_scan()
        method = getattr(self.control, 'reject_unknown_scan', None)
        self.assertTrue(callable(method), 'Controller must expose reject_unknown_scan')
        self.assertTrue(method())

    def opened(self):
        self.control.handle_mcu('RX,SERVO2,OPEN')
        return self.control.handle_mcu('STATUS,SORT_GATE_OPEN')

    def closed(self):
        self.control.handle_mcu('RX,SERVO2,CLOSE')
        return self.control.handle_mcu('STATUS,SORT_GATE_CLOSE')

    def test_open_close_and_settle_before_next_push_without_counting_success(self):
        self.begin_reject()
        epoch = self.control.epoch
        self.assertEqual(self.sent, ['SERVO2,OPEN'])
        self.assertEqual(self.control.snapshot().state, 'reject_opening')
        self.now = .2
        self.assertTrue(self.opened())
        self.now = 1.19
        self.control.tick()
        self.assertEqual(self.sent, ['SERVO2,OPEN'])
        self.now = 1.2
        self.control.tick()
        self.assertEqual(self.sent, ['SERVO2,OPEN', 'SERVO2,CLOSE'])
        self.assertEqual(self.control.snapshot().state, 'reject_closing')
        self.now = 1.4
        self.assertTrue(self.closed())
        self.now = 2.39
        self.control.tick()
        self.assertNotIn('PUSH,WAIT', self.sent)
        self.now = 2.4
        self.control.tick()
        self.assertEqual(self.sent, ['SERVO2,OPEN', 'SERVO2,CLOSE', 'PUSH,WAIT'])
        self.assertEqual(self.control.snapshot().state, 'wait_repush')
        self.assertGreater(self.control.epoch, epoch)
        self.assertTrue(self.control.handle_mcu('STATUS,PUSH,WAIT'))
        self.assertTrue(self.control.handle_mcu('PUSH?'))
        self.assertEqual(self.control.snapshot().state, 'locating')
        self.assertEqual(self.control.snapshot().completed_count, 0)
        self.assertFalse(self.control.handle_mcu('DONE'))
        self.assertFalse(any(line.startswith(('SORT,', 'BIN,')) for line in self.sent))

    def test_only_current_ready_origin_confirmed_scan_can_start_disposal(self):
        self.begin_reject()
        before = self.sent[:]
        self.assertFalse(self.control.reject_unknown_scan())
        self.assertEqual(self.sent, before)
        self.assertFalse(self.control.provide_sort(1))
        self.assertFalse(self.control.handle_mcu('PUSH?'))
        self.assertFalse(self.control.handle_mcu('SCAN'))
        self.assertEqual(self.control.snapshot().state, 'reject_opening')
        self.assertFalse(self.control.goto(80, 80))
        self.assertFalse(self.control.set_pusher(True))

    def test_expired_scan_never_opens_gate(self):
        self.reach_scan()
        method = getattr(self.control, 'reject_unknown_scan', None)
        self.assertTrue(callable(method), 'Controller must expose reject_unknown_scan')
        self.now = 5
        self.assertFalse(method())
        self.assertNotIn('SERVO2,OPEN', self.sent)
        self.assertIn('STOP', self.sent)

    def test_wrong_unproven_and_duplicate_replies_cannot_advance_timers(self):
        self.begin_reject()
        self.assertFalse(self.control.handle_mcu('STATUS,SORT_GATE_CLOSE'))
        self.assertFalse(self.control.handle_mcu('STATUS,SORT_GATE_OPEN'))
        self.assertEqual(self.control.snapshot().state, 'reject_opening')
        self.now = .1
        self.assertTrue(self.opened())
        self.now = .8
        self.assertFalse(self.opened())
        self.now = 1.1
        self.control.tick()
        self.assertEqual(self.sent[-1], 'SERVO2,CLOSE')
        self.assertFalse(self.control.handle_mcu('STATUS,SORT_GATE_CLOSE'))
        self.now = 1.2
        self.assertTrue(self.closed())
        self.now = 2
        self.assertFalse(self.closed())
        self.now = 2.2
        self.control.tick()
        self.assertEqual(self.sent.count('PUSH,WAIT'), 1)
        self.now = 2.3
        self.control.tick()
        self.assertEqual(self.sent.count('PUSH,WAIT'), 1)

    def test_open_ack_timeout_stops_and_requests_gate_close(self):
        self.begin_reject()
        self.now = 4.9
        self.control.handle_mcu('STATUS,MCU,WAIT_SORT,1960,195,123')
        self.now = 5
        self.control.tick()
        self.assertEqual(self.sent[-3:], ['SERVO2,CLOSE', 'STOP', 'STATUS?'])
        self.assertEqual(self.control.snapshot().state, 'stopped')
        self.assertFalse(self.control.ready)
        self.assertFalse(self.opened())
        self.now = 20
        self.control.tick()
        self.assertNotIn('PUSH,WAIT', self.sent)

    def test_late_ack_cannot_restart_expired_disposal(self):
        self.begin_reject()
        self.control.handle_mcu('RX,SERVO2,OPEN')
        self.now = 5
        self.assertFalse(self.control.handle_mcu('STATUS,SORT_GATE_OPEN'))
        self.assertEqual(self.control.snapshot().state, 'stopped')
        self.assertNotIn('PUSH,WAIT', self.sent)

    def test_close_ack_timeout_never_requests_another_push(self):
        self.begin_reject()
        self.opened()
        self.now = 1
        self.control.tick()
        self.now = 6
        self.control.tick()
        self.assertEqual(self.control.snapshot().state, 'stopped')
        self.assertIn('STOP', self.sent)
        self.assertNotIn('PUSH,WAIT', self.sent)

    def test_late_push_wait_ack_cannot_extend_resume_deadline(self):
        self.begin_reject()
        self.opened()
        self.now = 1
        self.control.tick()
        self.closed()
        self.now = 2
        self.control.tick()
        self.now = 32
        self.assertFalse(self.control.handle_mcu('STATUS,PUSH,WAIT'))
        self.assertEqual(self.control.snapshot().state, 'stopped')
        self.assertFalse(self.control.handle_mcu('PUSH?'))

    def test_stop_during_closed_gate_settling_never_resumes_push(self):
        self.begin_reject()
        self.opened()
        self.now = 1
        self.control.tick()
        self.closed()
        self.control.stop()
        self.now = 2
        self.control.tick()
        self.closed()
        self.assertNotIn('PUSH,WAIT', self.sent)

    def test_stop_closes_gate_and_cancels_pending_hold_and_resume(self):
        self.begin_reject()
        self.opened()
        self.assertTrue(self.control.stop())
        self.assertEqual(self.sent[-3:], ['SERVO2,CLOSE', 'STOP', 'STATUS?'])
        self.now = 2
        self.control.tick()
        self.opened()
        self.closed()
        self.assertNotIn('PUSH,WAIT', self.sent)
        self.assertFalse(self.control.snapshot().origin_confirmed)

    def test_mcu_stop_reply_also_closes_a_gate_opened_by_rejection(self):
        self.begin_reject()
        self.opened()
        self.assertTrue(self.control.handle_mcu('STATUS,STOP'))
        self.assertEqual(self.sent[-1], 'SERVO2,CLOSE')
        self.now = 2
        self.control.tick()
        self.assertEqual(self.control.snapshot().state, 'stopped')
        self.assertNotIn('PUSH,WAIT', self.sent)

    def test_disconnect_cancels_actions_and_reports_unconfirmed_gate(self):
        self.begin_reject()
        before = self.sent[:]
        self.control.disconnect('test serial lost')
        self.now = 2
        self.control.tick()
        self.assertEqual(self.sent, before)
        self.assertFalse(self.opened())
        self.assertFalse(self.closed())
        self.assertEqual(self.control.snapshot().state, 'disconnected')
        self.assertIn('底板', self.control.snapshot().last_error)

    def test_pause_after_rejection_does_not_push_another_part(self):
        self.begin_reject()
        self.assertTrue(self.control.set_feed_paused(True))
        self.opened()
        self.now = 1
        self.control.tick()
        self.closed()
        self.now = 2
        self.control.tick()
        self.control.handle_mcu('STATUS,PUSH,WAIT')
        self.control.handle_mcu('PUSH?')
        self.assertEqual(self.control.snapshot().state, 'paused')
        self.assertEqual(self.control.snapshot().completed_count, 0)
        self.assertTrue(self.control.snapshot().origin_confirmed)

    def test_reboot_during_disposal_cancels_gate_timers(self):
        self.begin_reject()
        self.opened()
        self.control.handle_mcu('BOOT,999')
        self.now = 2
        self.control.tick()
        self.assertEqual(self.control.snapshot().state, 'handshaking')
        self.assertNotIn('PUSH,WAIT', self.sent)
        self.assertFalse(self.control.snapshot().origin_confirmed)

    def test_unexpected_mcu_state_during_disposal_stops(self):
        self.begin_reject()
        self.control.handle_mcu('STATUS,MCU,WAIT_PUSH,1960,195,123')
        self.assertEqual(self.control.snapshot().state, 'stopped')
        self.assertNotIn('PUSH,WAIT', self.sent)


if __name__ == '__main__':
    unittest.main()
