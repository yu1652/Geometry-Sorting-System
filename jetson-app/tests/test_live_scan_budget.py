"""Regression for the Xavier 2026-11-16 21:23 scan timeout (no hardware)."""
import unittest

from live_control import LiveController
from linux_20261107 import SCAN_ATTEMPTS, SCAN_TIMEOUT, DONE_TIMEOUT
from scan_gate import ScanConsensus


class ScanBudgetTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.sent = []
        self.machine = LiveController(self.sent.append, clock=lambda: self.now,
                                      scan_timeout=SCAN_TIMEOUT, done_timeout=DONE_TIMEOUT)
        self.machine.begin_handshake()
        for line in ('PONG', 'VERSION,ESP32_SORTER_XY_PUSHER,0.1.0',
                     'STATUS,MCU,WAIT_START,0,0,77'):
            self.machine.handle_mcu(line)
        self.assertTrue(self.machine.confirm_origin())
        self.machine.handle_mcu('READY')
        self.assertTrue(self.machine.start())
        self.machine.handle_mcu('STATUS,HOST_READY')
        self.machine.handle_mcu('STATUS,START')
        self.machine.handle_mcu('PUSH?')
        self.machine.provide_target(127.172, 120.092)
        self.machine.handle_mcu('SCAN')
        self.sent.clear()

    def heartbeat(self, now):
        self.now = now
        self.machine.handle_mcu('STATUS,MCU,WAIT_SORT,1960,195,77')
        self.machine.tick()

    def test_recorded_three_observations_can_finish_with_scheduling_margin(self):
        # Log: empty, yellow cylinder, same cylinder; final decision at 5.004 s.
        # Also exercise slower capture/storage on a busy Xavier (up to 12 s).
        for duration in (5.004, 9.0, 12.0):
            with self.subTest(duration=duration):
                self.setUp()
                gate = ScanConsensus(max_attempts=SCAN_ATTEMPTS, required_matches=2)
                for elapsed, key, decision in (
                        (1.487, None, 'retry'),
                        (3.269, ('yuan_zhu', '黄色'), 'retry'),
                        (duration, ('yuan_zhu', '黄色'), 'accept')):
                    self.heartbeat(elapsed)
                    self.assertEqual(gate.observe(key), decision)
                    self.assertEqual(self.machine.snapshot().state, 'recognizing')
                self.assertTrue(self.machine.provide_sort(1))
                self.assertEqual(self.sent, ['SORT,1'])
                self.assertEqual(self.machine.snapshot().completed_count, 0)
                self.assertTrue(self.machine.handle_mcu('DONE'))
                self.assertEqual(self.machine.snapshot().completed_count, 1)

    def test_scan_deadline_remains_bounded_despite_healthy_heartbeats(self):
        self.assertLessEqual(SCAN_TIMEOUT, 15.0)
        for seconds in range(1, int(SCAN_TIMEOUT)):
            self.heartbeat(seconds)
            self.assertEqual(self.machine.snapshot().state, 'recognizing')
        self.heartbeat(SCAN_TIMEOUT)
        self.assertEqual(self.sent, ['STOP', 'STATUS?'])
        self.assertIn('识别复核超时', self.machine.snapshot().last_error)
        self.assertFalse(self.machine.provide_sort(1))
        self.assertNotIn('SORT,1', self.sent)
        self.assertEqual(self.machine.snapshot().completed_count, 0)

    def test_late_callback_without_timer_poll_has_same_scan_error(self):
        self.now = SCAN_TIMEOUT
        self.assertFalse(self.machine.provide_sort(1))
        self.assertIn('识别复核超时', self.machine.snapshot().last_error)
        self.assertEqual(self.sent, ['STOP', 'STATUS?'])

    def test_drop_watchdog_is_not_extended_with_scan_budget(self):
        self.assertTrue(self.machine.provide_sort(1))
        self.now = DONE_TIMEOUT
        self.assertFalse(self.machine.handle_mcu('DONE'))
        self.assertIn('投箱完成等待超时', self.machine.snapshot().last_error)
        self.assertEqual(self.machine.snapshot().completed_count, 0)

    def test_missing_serial_responses_are_reported_as_communication_timeout(self):
        self.now = 30.0
        self.machine.tick()
        self.assertIn('串口通信超时', self.machine.snapshot().last_error)
        self.assertFalse(self.machine.ready)
        self.assertFalse(self.machine.snapshot().origin_confirmed)


if __name__ == '__main__':
    unittest.main()
