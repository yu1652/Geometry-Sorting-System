import unittest

from sorter_simulation import SimulatedSorter


class SorterSimulationTests(unittest.TestCase):
    def setUp(self):
        self.sorter = SimulatedSorter()

    def start_sorter(self):
        self.assertTrue(self.sorter.confirm_origin())
        self.assertTrue(self.sorter.start())

    def test_requires_origin_and_stop_requires_reconfirmation(self):
        self.assertFalse(self.sorter.start())
        self.assertFalse(self.sorter.goto(20, 30))
        self.assertFalse(self.sorter.set_pusher(True))
        self.assertFalse(self.sorter.run_cycle(20, 30, 1))
        self.start_sorter()
        self.assertTrue(self.sorter.stop())
        self.assertFalse(self.sorter.start())
        self.assertFalse(self.sorter.run_cycle(20, 30, 1))
        self.assertFalse(self.sorter.handle_mcu("DONE"))
        self.assertFalse(self.sorter.snapshot().origin_confirmed)
        self.start_sorter()
        self.assertEqual(self.sorter.snapshot().completed_count, 0)

    def test_completes_cycle_only_once(self):
        self.start_sorter()
        self.assertFalse(self.sorter.handle_mcu("SCAN"))
        self.assertFalse(self.sorter.handle_mcu("DONE"))
        self.assertTrue(self.sorter.run_cycle(123.4, 55, 2))
        self.assertFalse(self.sorter.handle_mcu("DONE"))
        snapshot = self.sorter.snapshot()
        self.assertEqual(snapshot.state, "running")
        self.assertEqual(snapshot.completed_count, 1)
        self.assertEqual(snapshot.bin_counts, (0, 1, 0, 0, 0, 0))
        self.assertIn("HOST PUSH,123.4,55.0,mm", snapshot.log)
        self.assertIn("HOST SORT,2", snapshot.log)

    def test_maintenance_requires_raised_pusher_and_disarmed_cycle(self):
        self.assertTrue(self.sorter.confirm_origin())
        self.assertTrue(self.sorter.goto(20, 30))
        self.assertTrue(self.sorter.set_pusher(True))
        self.assertFalse(self.sorter.start())
        self.assertFalse(self.sorter.goto(30, 40))
        self.assertTrue(self.sorter.set_pusher(False))
        self.assertTrue(self.sorter.start())
        self.assertFalse(self.sorter.set_pusher(True))
        self.assertFalse(self.sorter.goto(30, 40))
        self.assertTrue(self.sorter.set_pusher(False))
        self.assertEqual((self.sorter.snapshot().x_mm, self.sorter.snapshot().y_mm), (20, 30))

    def test_outside_sample_envelope_and_nonfinite_motion_are_rejected(self):
        self.assertTrue(self.sorter.confirm_origin())
        for x, y in [(float("nan"), 30), (20, float("inf")), (-1, 20), (161, 20), (True, 20)]:
            with self.subTest(x=x, y=y):
                self.assertFalse(self.sorter.goto(x, y))
        self.assertTrue(self.sorter.start())
        for x in [10, 10.01, 161, float("nan")]:
            self.assertFalse(self.sorter.run_cycle(x, 20, 1))
        self.assertEqual(self.sorter.snapshot().state, "running")
        self.assertEqual(self.sorter.snapshot().completed_count, 0)

    def test_reboot_and_error_invalidate_origin(self):
        self.start_sorter()
        self.assertTrue(self.sorter.handle_mcu("STATUS,MCU,WAIT_PUSH,0,0,123"))
        self.assertTrue(self.sorter.snapshot().origin_confirmed)
        self.assertTrue(self.sorter.handle_mcu("STATUS,MCU,WAIT_START,0,0,124"))
        self.assertFalse(self.sorter.snapshot().origin_confirmed)
        self.assertFalse(self.sorter.handle_mcu("DONE"))
        self.start_sorter()
        self.assertTrue(self.sorter.handle_mcu("ERROR,AXIS"))
        self.assertEqual(self.sorter.snapshot().state, "stopped")
        self.assertEqual(self.sorter.snapshot().last_error, "ERROR,AXIS")
        self.assertFalse(self.sorter.snapshot().origin_confirmed)

    def test_invalid_bin_stops_without_counting(self):
        self.start_sorter()
        self.assertFalse(self.sorter.run_cycle(20, 30, True))
        self.assertEqual(self.sorter.snapshot().completed_count, 0)
        self.assertEqual(self.sorter.snapshot().state, "stopped")

    def test_duplicate_push_replays_plan_and_stopped_cycle_ignores_late_done(self):
        self.start_sorter()
        self.assertTrue(self.sorter.begin_cycle(20, 30, 4))
        self.assertTrue(self.sorter.handle_mcu("PUSH?"))
        self.assertEqual(self.sorter.snapshot().log.count("HOST PUSH,20.0,30.0,mm"), 2)
        self.assertFalse(self.sorter.handle_mcu("DONE"))
        self.assertTrue(self.sorter.handle_mcu("SCAN"))
        self.assertFalse(self.sorter.handle_mcu("SCAN"))
        self.assertTrue(self.sorter.stop())
        self.assertFalse(self.sorter.handle_mcu("DONE"))
        self.assertEqual(self.sorter.snapshot().completed_count, 0)

    def test_push_without_target_waits_but_invalid_target_reports_ng(self):
        self.start_sorter()
        self.assertTrue(self.sorter.handle_mcu("PUSH?"))
        self.assertEqual(self.sorter.snapshot().log[-1], "HOST PUSH,WAIT")
        self.assertFalse(self.sorter.run_cycle(10.01, 20, 1))
        self.assertIn("HOST PUSH,NG", self.sorter.snapshot().log)
        self.assertEqual(self.sorter.snapshot().state, "running")
