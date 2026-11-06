import unittest


class MotionLimitsTests(unittest.TestCase):
    def test_machine_region_accepts_inclusive_boundaries_above_tray(self):
        from motion_limits import position_in_machine
        for x, y in ((0, 0), (260, 230), (260, 0), (0, 230), (161, 161)):
            with self.subTest(x=x, y=y):
                self.assertTrue(position_in_machine(x, y))

    def test_invalid_coordinates_are_rejected_on_each_axis(self):
        from motion_limits import position_in_machine
        for invalid in (True, False, None, '1', float('nan'), float('inf'),
                        float('-inf'), -0.001, 10 ** 400):
            with self.subTest(invalid=invalid):
                self.assertFalse(position_in_machine(invalid, 1))
                self.assertFalse(position_in_machine(1, invalid))
        self.assertFalse(position_in_machine(260.001, 1))
        self.assertFalse(position_in_machine(1, 230.001))
