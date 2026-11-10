import unittest

from live_control import LiveController


class LiveControllerTests(unittest.TestCase):
    def setUp(self):
        self.sent = []
        self.now = 0.0
        self.control = LiveController(self.sent.append, clock=lambda: self.now, timeout=30)

    def handshake(self):
        self.control.begin_handshake()
        for line in ('PONG', 'VERSION,ESP32_SORTER_XY_PUSHER,0.1.0',
                     'STATUS,MCU,WAIT_START,0,0,123'):
            self.control.handle_mcu(line)
        self.assertTrue(self.control.ready)

    def arm(self):
        self.handshake()
        self.assertTrue(self.control.confirm_origin())
        self.assertEqual(self.sent[-1], 'RESET')
        self.control.handle_mcu('STATUS,PUSHER_UP')
        self.control.handle_mcu('READY')
        self.assertNotIn('HOST_READY', self.sent)
        self.assertTrue(self.control.snapshot().origin_confirmed)

    def start_running(self):
        self.arm()
        self.assertTrue(self.control.start())
        self.assertEqual(self.sent[-1], 'HOST_READY')
        self.control.handle_mcu('STATUS,HOST_READY')
        self.assertEqual(self.sent[-1], 'START')
        self.control.handle_mcu('STATUS,START')
        self.assertEqual(self.control.snapshot().state, 'running')

    def test_handshake_requires_all_three_real_replies_and_new_status(self):
        self.control.begin_handshake()
        self.assertEqual(self.sent, ['PING', 'VERSION?', 'STATUS?'])
        self.control.handle_mcu('RX,PING')
        self.control.handle_mcu('VERSION,ESP32_SORTER_XY_PUSHER,0.1.0')
        self.control.handle_mcu('STATUS,MCU,WAIT_START,0,0')
        self.assertFalse(self.control.ready)
        self.control.handle_mcu('PONG')
        self.assertFalse(self.control.ready)
        self.control.handle_mcu('STATUS,MCU,WAIT_START,0,0,123')
        self.assertTrue(self.control.ready)
        self.assertFalse(self.control.snapshot().origin_confirmed)

    def test_no_motion_before_confirmed_origin_and_real_reset_ack(self):
        self.assertFalse(self.control.confirm_origin())
        self.assertFalse(self.control.start())
        self.assertFalse(self.control.goto(10, 20))
        self.assertFalse(self.control.set_pusher(True))
        self.assertEqual(self.sent, [])
        self.handshake()
        self.control.confirm_origin()
        self.assertFalse(self.control.start())
        self.control.handle_mcu('RX,RESET')
        self.assertFalse(self.control.snapshot().origin_confirmed)
        self.control.handle_mcu('READY')
        self.assertTrue(self.control.snapshot().origin_confirmed)
        self.assertNotIn('HOST_READY', self.sent)
        self.control.handle_mcu('STATUS,HOST_READY')
        self.assertTrue(self.control.snapshot().origin_confirmed)
        self.assertNotIn('START', self.sent)

    def test_goto_uses_pulse_status_and_excludes_other_motion_until_ack(self):
        self.arm()
        self.assertTrue(self.control.goto(20, 30))
        self.assertEqual(self.sent[-2:], ['GOTO_MM,20.000,30.000', 'STATUS?'])
        self.assertTrue(self.control.pending)
        self.assertFalse(self.control.goto(30, 40))
        self.assertFalse(self.control.set_pusher(True))
        self.assertFalse(self.control.start())
        self.control.handle_mcu('RX,GOTO_MM,20.000,30.000')
        self.control.handle_mcu('STATUS,PUSHER_UP')
        self.assertTrue(self.control.pending)
        self.control.handle_mcu('STATUS,MCU,WAIT_START,800,1170,123')
        self.assertFalse(self.control.pending)
        self.assertEqual((self.control.snapshot().x_mm, self.control.snapshot().y_mm), (20, 30))

    def test_machine_corner_goto_waits_for_exact_real_pulses(self):
        self.arm()
        self.assertTrue(self.control.goto(260, 230))
        self.assertEqual(self.sent[-2:], ['GOTO_MM,260.000,230.000', 'STATUS?'])
        for line in ('RX,GOTO_MM,260.000,230.000', 'STATUS,PUSHER_UP',
                     'STATUS,MCU,WAIT_START,10400,8969,123',
                     'STATUS,MCU,WAIT_START,10399,8970,123',
                     'STATUS,MCU,IDLE,10400,8970,123'):
            self.control.handle_mcu(line)
            self.assertEqual(self.control.pending, 'goto')
        self.control.handle_mcu('STATUS,MCU,WAIT_START,10400,8970,123')
        self.assertIsNone(self.control.pending)
        self.assertEqual((self.control.snapshot().x_mm, self.control.snapshot().y_mm), (260, 230))

    def test_push_accepts_machine_corner_above_tray_size(self):
        self.start_running()
        self.control.handle_mcu('PUSH?')
        self.assertTrue(self.control.provide_target(260, 230))
        self.assertEqual(self.sent[-1], 'PUSH,260.000,230.000,mm')
        self.assertEqual(self.control.snapshot().state, 'wait_scan')

    def test_push_beyond_either_machine_axis_stops_without_push(self):
        for x, y in ((260.001, 230), (260, 230.001)):
            with self.subTest(x=x, y=y):
                self.setUp()
                self.start_running()
                self.control.handle_mcu('PUSH?')
                before = len(self.sent)
                self.assertFalse(self.control.provide_target(x, y))
                self.assertEqual(self.sent[before:], ['STOP', 'STATUS?'])
                self.assertFalse(self.control.snapshot().origin_confirmed)

    def test_lift_waits_for_real_ack_and_up_can_cancel_pending_down(self):
        self.arm()
        self.assertTrue(self.control.set_pusher(True))
        self.assertFalse(self.control.snapshot().pusher_down)
        self.assertFalse(self.control.goto(10, 10))
        self.assertTrue(self.control.set_pusher(False))
        self.control.handle_mcu('STATUS,PUSHER_DOWN')
        self.assertTrue(self.control.pending)
        self.assertTrue(self.control.snapshot().pusher_down)
        self.control.handle_mcu('STATUS,PUSHER_UP')
        self.assertFalse(self.control.pending)
        self.assertTrue(self.control.goto(10, 10))

    def test_automatic_cycle_needs_real_events_and_counts_done_once(self):
        self.start_running()
        epoch = self.control.epoch
        self.control.handle_mcu('PUSH?')
        self.assertEqual(self.control.snapshot().state, 'locating')
        self.assertGreater(self.control.epoch, epoch)
        epoch = self.control.epoch
        self.control.handle_mcu('PUSH?')
        self.assertEqual(self.control.epoch, epoch)
        self.assertTrue(self.control.provide_target(100, 50))
        self.assertEqual(self.sent[-1], 'PUSH,100.000,50.000,mm')
        self.assertEqual(self.control.snapshot().completed_count, 0)
        self.assertFalse(self.control.handle_mcu('DONE'))
        self.control.handle_mcu('SCAN')
        self.assertTrue(self.control.provide_sort(2))
        self.assertEqual(self.sent[-1], 'SORT,2')
        self.assertEqual(self.control.snapshot().completed_count, 0)
        self.assertTrue(self.control.handle_mcu('DONE'))
        self.assertFalse(self.control.handle_mcu('DONE'))
        self.assertEqual(self.control.snapshot().completed_count, 1)
        self.assertEqual(self.control.snapshot().bin_counts, (0, 1, 0, 0, 0, 0))

    def test_optional_drop_verification_holds_push_and_counts_once(self):
        self.control = LiveController(self.sent.append, clock=lambda: self.now,
                                      timeout=30, verify_drop=True)
        self.start_running()
        self.control.handle_mcu('PUSH?')
        self.control.provide_target(100, 50)
        self.control.handle_mcu('SCAN')
        self.control.provide_sort(2)
        before = list(self.sent)
        self.assertTrue(self.control.handle_mcu('DONE'))
        epoch = self.control.request_epoch
        self.assertEqual(self.control.snapshot().state, 'verifying_drop')
        self.assertEqual(self.control.snapshot().completed_count, 0)
        self.assertFalse(self.control.handle_mcu('DONE'))
        self.assertFalse(self.control.handle_mcu('PUSH?'))
        self.assertEqual(self.sent, before)
        self.assertEqual(self.control.request_epoch, epoch)
        self.assertTrue(self.control.confirm_drop(epoch))
        self.assertFalse(self.control.confirm_drop(epoch))
        self.assertEqual(self.control.snapshot().bin_counts, (0, 1, 0, 0, 0, 0))
        self.assertEqual(self.control.snapshot().state, 'running')
        self.assertEqual(self.sent, before)
        self.assertTrue(self.control.handle_mcu('PUSH?'))
        self.assertEqual(self.control.snapshot().state, 'locating')

    def test_drop_verification_requires_fresh_epoch_and_expires(self):
        self.control = LiveController(self.sent.append, clock=lambda: self.now,
                                      timeout=30, verify_drop=True)
        self.start_running()
        self.control.handle_mcu('PUSH?')
        self.control.provide_target(100, 50)
        self.control.handle_mcu('SCAN')
        stale_epoch = self.control.request_epoch
        self.control.provide_sort(2)
        self.now = 29
        self.control.handle_mcu('DONE')
        self.assertFalse(self.control.confirm_drop(stale_epoch))
        self.assertEqual(self.control.snapshot().completed_count, 0)
        self.now = 31
        self.control.tick()
        self.assertEqual(self.control.snapshot().state, 'verifying_drop')
        self.now = 60
        self.control.tick()
        self.assertFalse(self.control.confirm_drop(self.control.request_epoch))
        self.assertEqual(self.control.snapshot().completed_count, 0)
        self.assertFalse(self.control.snapshot().origin_confirmed)

    def test_previous_drop_epoch_cannot_confirm_next_drop(self):
        self.control = LiveController(self.sent.append, clock=lambda: self.now,
                                      timeout=30, verify_drop=True)
        self.start_running()
        for bin_id in (2, 3):
            self.control.handle_mcu('PUSH?')
            self.control.provide_target(100, 50)
            self.control.handle_mcu('SCAN')
            self.control.provide_sort(bin_id)
            self.control.handle_mcu('DONE')
            if bin_id == 2:
                previous_epoch = self.control.request_epoch
                self.assertTrue(self.control.confirm_drop(previous_epoch))
        self.assertFalse(self.control.confirm_drop(previous_epoch))
        self.assertEqual(self.control.snapshot().completed_count, 1)
        self.assertTrue(self.control.confirm_drop(self.control.request_epoch))
        self.assertEqual(self.control.snapshot().bin_counts, (0, 1, 1, 0, 0, 0))

    def test_drop_verification_invalidated_by_stop_boot_and_error(self):
        for invalidation in ('stop', 'BOOT,124', 'ERROR,AXIS'):
            with self.subTest(invalidation=invalidation):
                self.setUp()
                self.control = LiveController(self.sent.append, clock=lambda: self.now,
                                              timeout=30, verify_drop=True)
                self.start_running()
                self.control.handle_mcu('PUSH?')
                self.control.provide_target(100, 50)
                self.control.handle_mcu('SCAN')
                self.control.provide_sort(2)
                self.control.handle_mcu('DONE')
                epoch = self.control.request_epoch
                if invalidation == 'stop':
                    self.control.stop()
                else:
                    self.control.handle_mcu(invalidation)
                self.assertFalse(self.control.confirm_drop(epoch))
                self.assertEqual(self.control.snapshot().completed_count, 0)

    def test_optional_scan_timeout_only_applies_to_recognizing(self):
        self.control = LiveController(self.sent.append, clock=lambda: self.now,
                                      timeout=30, scan_timeout=5)
        self.start_running()
        self.control.handle_mcu('PUSH?')
        self.control.provide_target(100, 50)
        self.now = 10
        self.control.handle_mcu('SCAN')
        self.now = 14
        self.control.tick()
        self.assertEqual(self.control.snapshot().state, 'recognizing')
        self.now = 15
        self.control.tick()
        self.assertFalse(self.control.snapshot().origin_confirmed)

    def test_late_sort_result_cannot_extend_expired_scan_deadline(self):
        self.control = LiveController(self.sent.append, clock=lambda: self.now,
                                      timeout=30, scan_timeout=5)
        self.start_running()
        self.control.handle_mcu('PUSH?')
        self.control.provide_target(100, 50)
        self.control.handle_mcu('SCAN')
        self.now = 5.01
        self.assertFalse(self.control.provide_sort(2))
        self.assertNotIn('SORT,2', self.sent)
        self.assertFalse(self.control.snapshot().origin_confirmed)

    def test_late_mcu_transition_cannot_extend_expired_deadline(self):
        for late_line, setup in (
            ('SCAN', 'wait_scan'), ('DONE', 'wait_done'),
            ('PUSH?', 'running'), ('READY', 'resetting'),
        ):
            with self.subTest(late_line=late_line):
                self.setUp()
                self.control = LiveController(self.sent.append, clock=lambda: self.now,
                                              timeout=30, verify_drop=True)
                if setup == 'resetting':
                    self.handshake()
                    self.control.confirm_origin()
                else:
                    self.start_running()
                    if setup != 'running':
                        self.control.handle_mcu('PUSH?')
                        self.control.provide_target(100, 50)
                    if setup == 'wait_done':
                        self.control.handle_mcu('SCAN')
                        self.control.provide_sort(2)
                self.now = 31
                self.assertFalse(self.control.handle_mcu(late_line))
                self.assertFalse(self.control.snapshot().origin_confirmed)
                self.assertEqual(self.control.snapshot().completed_count, 0)

    def test_late_drop_confirmation_fails_before_tick(self):
        self.control = LiveController(self.sent.append, clock=lambda: self.now,
                                      timeout=30, verify_drop=True)
        self.start_running()
        self.control.handle_mcu('PUSH?')
        self.control.provide_target(100, 50)
        self.control.handle_mcu('SCAN')
        self.control.provide_sort(2)
        self.control.handle_mcu('DONE')
        epoch = self.control.request_epoch
        self.now = 30.01
        self.assertFalse(self.control.confirm_drop(epoch))
        self.assertFalse(self.control.snapshot().origin_confirmed)
        self.assertEqual(self.control.snapshot().completed_count, 0)

    def test_late_host_ready_ack_cannot_send_start(self):
        self.control = LiveController(self.sent.append, clock=lambda: self.now,
                                      timeout=30, verify_drop=True)
        self.arm()
        self.assertTrue(self.control.start())
        self.now = 30.01
        self.assertFalse(self.control.handle_mcu('STATUS,HOST_READY'))
        self.assertNotIn('START', self.sent)
        self.assertFalse(self.control.snapshot().origin_confirmed)
        self.assertFalse(self.control.ready)

    def test_no_target_waits_and_next_request_has_new_epoch(self):
        self.start_running()
        self.control.handle_mcu('PUSH?')
        epoch = self.control.epoch
        self.assertTrue(self.control.provide_target(None))
        self.assertEqual(self.sent[-1], 'PUSH,WAIT')
        self.control.handle_mcu('PUSH?')
        self.assertGreater(self.control.epoch, epoch)

    def test_confirmed_empty_station_requests_next_push_without_done_or_sort(self):
        self.start_running()
        before = list(self.sent)
        self.assertFalse(self.control.retry_empty_scan())
        self.assertEqual(self.sent, before)
        self.control.handle_mcu('PUSH?')
        self.control.provide_target(100, 50)
        self.control.handle_mcu('SCAN')
        epoch = self.control.epoch
        self.assertTrue(self.control.retry_empty_scan())
        self.assertEqual(self.sent[-1], 'PUSH,WAIT')
        self.assertEqual(self.control.snapshot().state, 'wait_repush')
        self.assertGreater(self.control.epoch, epoch)
        self.assertFalse(self.control.provide_sort(1))
        self.assertFalse(self.control.handle_mcu('DONE'))
        self.assertTrue(self.control.handle_mcu('STATUS,PUSH,WAIT'))
        self.assertEqual(self.control.snapshot().state, 'running')
        self.assertTrue(self.control.handle_mcu('PUSH?'))
        self.assertEqual(self.control.snapshot().state, 'locating')
        self.assertEqual(self.control.snapshot().completed_count, 0)
        self.assertTrue(self.control.snapshot().origin_confirmed)

    def test_empty_retry_can_use_real_next_push_as_ack_but_stop_never_resumes(self):
        self.start_running()
        self.control.handle_mcu('PUSH?')
        self.control.provide_target(100, 50)
        self.control.handle_mcu('SCAN')
        self.assertTrue(self.control.retry_empty_scan())
        self.assertTrue(self.control.handle_mcu('PUSH?'))
        self.assertEqual(self.control.snapshot().state, 'locating')
        self.control.stop()
        self.assertFalse(self.control.handle_mcu('STATUS,PUSH,WAIT'))
        self.assertFalse(self.control.handle_mcu('PUSH?'))
        self.assertFalse(self.control.retry_empty_scan())
        self.assertEqual(self.control.snapshot().state, 'stopped')

    def test_stop_reboot_error_and_disconnect_invalidate_origin_and_jobs(self):
        self.start_running()
        self.control.handle_mcu('PUSH?')
        self.control.provide_target(100, 50)
        self.control.handle_mcu('SCAN')
        epoch = self.control.epoch
        self.assertTrue(self.control.stop())
        self.assertGreater(self.control.epoch, epoch)
        self.assertFalse(self.control.provide_sort(2))
        self.assertFalse(self.control.handle_mcu('DONE'))
        self.assertFalse(self.control.snapshot().origin_confirmed)
        self.control.handle_mcu('STATUS,HOST_READY')
        self.assertFalse(self.control.snapshot().origin_confirmed)
        self.control.handle_mcu('BOOT,124')
        self.assertFalse(self.control.ready)
        self.assertFalse(self.control.start())
        self.control.disconnect('USB unplugged')
        before = list(self.sent)
        self.assertFalse(self.control.stop())
        self.assertFalse(self.control.set_pusher(False))
        self.assertEqual(before, self.sent)

    def test_boot_id_change_and_error_never_resume(self):
        self.start_running()
        self.control.handle_mcu('STATUS,MCU,WAIT_START,0,0,124')
        self.assertFalse(self.control.snapshot().origin_confirmed)
        self.assertFalse(self.control.ready)
        self.assertFalse(self.control.provide_target(100, 40))
        self.control.handle_mcu('ERROR,AXIS')
        self.assertFalse(self.control.snapshot().origin_confirmed)
        self.assertIn('ERROR,AXIS', self.control.snapshot().last_error)

    def test_timeout_and_write_failure_disarm(self):
        self.arm()
        self.control.goto(20, 30)
        self.now = 31
        self.control.tick()
        self.assertFalse(self.control.snapshot().origin_confirmed)
        self.assertEqual(self.sent[-2:], ['STOP', 'STATUS?'])
        broken = LiveController(lambda line: (_ for _ in ()).throw(OSError('USB failed')))
        self.assertFalse(broken.begin_handshake())
        self.assertFalse(broken.connected)

    def test_bad_coordinates_and_bins_do_not_send_motion(self):
        self.arm()
        for x, y in ((-1, 0), (260.001, 0), (0, 230.001), (0, -1), (float('nan'), 1), (1, float('inf')), (True, 1)):
            before = list(self.sent)
            self.assertFalse(self.control.goto(x, y))
            self.assertEqual(self.sent, before)
        self.control.start()
        self.control.handle_mcu('STATUS,HOST_READY')
        self.control.handle_mcu('STATUS,START')
        self.control.handle_mcu('PUSH?')
        self.assertFalse(self.control.provide_target(10, 20))
        self.assertNotIn('PUSH,10.000,20.000,mm', self.sent)

    def test_stop_after_pending_down_does_not_accept_late_lift_as_origin(self):
        self.arm()
        self.control.set_pusher(True)
        self.control.stop()
        self.control.handle_mcu('STATUS,PUSHER_DOWN')
        self.control.handle_mcu('STATUS,PUSHER_UP')
        self.assertFalse(self.control.snapshot().origin_confirmed)
        self.assertFalse(self.control.start())

    def test_invalid_bin_stops_without_sort_or_counting(self):
        self.start_running()
        self.control.handle_mcu('PUSH?')
        self.control.provide_target(100, 30)
        self.control.handle_mcu('SCAN')
        self.assertFalse(self.control.provide_sort(True))
        self.assertFalse(any(line.startswith('SORT,') for line in self.sent))
        self.assertEqual(self.control.snapshot().completed_count, 0)
        self.assertFalse(self.control.snapshot().origin_confirmed)

    def test_stale_status_cannot_finish_goto_and_missing_arrival_times_out(self):
        self.arm()
        self.control.goto(20, 30)
        self.control.handle_mcu('STATUS,MCU,WAIT_START,0,0,123')
        self.assertEqual(self.control.pending, 'goto')
        self.now = 31
        self.control.tick()
        self.assertFalse(self.control.snapshot().origin_confirmed)
        self.assertIn('STOP', self.sent)

    def test_manual_wire_rounding_and_nonrepeating_timeout(self):
        self.arm()
        self.control.goto(20, 0.0128)
        self.control.handle_mcu('STATUS,MCU,WAIT_START,800,1,123')
        self.assertFalse(self.control.pending)
        self.control.goto(30, 30)
        self.now = 31
        self.control.tick()
        before = list(self.sent)
        self.control.tick()
        self.assertEqual(self.sent, before)

    def test_log_is_bounded_and_disconnected_late_replies_are_ignored(self):
        self.handshake()
        for _ in range(300):
            self.control.handle_mcu('PONG')
        self.assertEqual(len(self.control.snapshot().log), 200)
        self.control.disconnect()
        self.control.handle_mcu('READY')
        self.control.handle_mcu('STATUS,HOST_READY')
        self.assertFalse(self.control.ready)
        self.assertFalse(self.control.snapshot().origin_confirmed)

    def test_heartbeat_does_not_extend_an_action_deadline(self):
        self.arm()
        self.control.goto(20, 30)
        self.now = 29
        self.control.handle_mcu('PONG')
        self.now = 31
        self.control.tick()
        self.assertFalse(self.control.snapshot().origin_confirmed)

    def test_reboot_handshake_requires_fresh_identity_replies(self):
        self.start_running()
        self.control.handle_mcu('BOOT,124')
        self.control.handle_mcu('STATUS,MCU,WAIT_START,0,0,124')
        self.assertFalse(self.control.ready)
        self.control.handle_mcu('PONG')
        self.assertFalse(self.control.ready)
        self.control.handle_mcu('VERSION,ESP32_SORTER_XY_PUSHER,0.1.0')
        self.assertTrue(self.control.ready)
        self.assertFalse(self.control.snapshot().origin_confirmed)

    def test_manual_origin_does_not_enable_physical_start_and_unexpected_start_stops(self):
        self.arm()
        self.assertFalse(any(line in ('HOST_READY', 'START') for line in self.sent))
        self.assertFalse(self.control.handle_mcu('STATUS,START'))
        self.assertFalse(self.control.snapshot().origin_confirmed)
        self.assertEqual(self.sent[-2:], ['STOP', 'STATUS?'])
        self.assertFalse(self.control.provide_target(100, 30))

    def test_auto_requires_ordered_acks_and_duplicate_ack_cannot_start_twice(self):
        self.arm()
        self.control.start()
        self.assertEqual(self.control.snapshot().state, 'enabling_auto')
        self.assertFalse(self.control.handle_mcu('PUSH?'))
        self.assertNotIn('START', self.sent)
        self.control.handle_mcu('STATUS,HOST_READY')
        self.assertEqual(self.control.snapshot().state, 'starting')
        self.control.handle_mcu('STATUS,HOST_READY')
        self.assertEqual(self.sent.count('START'), 1)
        self.control.handle_mcu('STATUS,START')
        self.assertEqual(self.control.snapshot().state, 'running')
        self.control.stop()
        self.control.handle_mcu('STATUS,HOST_READY')
        self.assertEqual(self.sent.count('START'), 1)
        self.assertFalse(self.control.snapshot().origin_confirmed)

    def test_stop_requires_mcu_stop_ack_before_reconfirming_origin(self):
        self.arm()
        self.control.stop()
        self.assertFalse(self.control.confirm_origin())
        self.control.handle_mcu('STATUS,STOP')
        self.assertTrue(self.control.confirm_origin())
        self.control.handle_mcu('READY')
        self.assertTrue(self.control.snapshot().origin_confirmed)
        self.assertEqual(self.sent[-1], 'RESET')
        self.assertFalse(any(line == 'HOST_READY' for line in self.sent))

    def test_duplicate_ready_invalidates_manual_origin(self):
        self.arm()
        self.control.handle_mcu('READY')
        self.assertFalse(self.control.snapshot().origin_confirmed)
        self.assertFalse(self.control.ready)
        self.assertFalse(self.control.goto(20, 30))


if __name__ == '__main__':
    unittest.main()
