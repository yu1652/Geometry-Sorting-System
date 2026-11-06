"""Tests for bounded consecutive scan evidence."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scan_gate import ScanConsensus


class ScanConsensusTests(unittest.TestCase):
    def test_accepts_only_after_two_consecutive_equal_keys(self):
        gate = ScanConsensus()
        self.assertEqual((gate.attempts, gate.matches, gate.candidate), (0, 0, None))
        self.assertEqual(gate.observe(('circle', 'red')), 'retry')
        self.assertEqual((gate.attempts, gate.matches, gate.candidate), (1, 1, ('circle', 'red')))
        self.assertEqual(gate.observe(('circle', 'red')), 'accept')
        self.assertEqual((gate.attempts, gate.matches), (2, 2))

    def test_changed_shape_or_color_and_unavailable_reset_streak(self):
        gate = ScanConsensus()
        for key in [('circle', 'red'), ('square', 'red'), ('square', 'blue')]:
            self.assertEqual(gate.observe(key), 'retry')
            self.assertEqual(gate.matches, 1)
            self.assertEqual(gate.candidate, key)
        self.assertEqual(gate.observe(None), 'retry')
        self.assertEqual((gate.matches, gate.candidate), (0, None))
        self.assertEqual(gate.observe(('square', 'blue')), 'retry')
        self.assertEqual(gate.observe(('square', 'blue')), 'accept')

    def test_budget_exhausts_after_eight_unavailable_samples(self):
        gate = ScanConsensus()
        for _ in range(7):
            self.assertEqual(gate.observe(None), 'retry')
        self.assertEqual(gate.observe(None), 'exhausted')
        self.assertEqual((gate.attempts, gate.matches, gate.candidate), (8, 0, None))

    def test_final_attempt_can_accept_but_cannot_start_an_extra_attempt(self):
        gate = ScanConsensus(max_attempts=3)
        self.assertEqual(gate.observe(None), 'retry')
        self.assertEqual(gate.observe(('circle', 'red')), 'retry')
        self.assertEqual(gate.observe(('circle', 'red')), 'accept')
        exhausted = ScanConsensus(max_attempts=2)
        self.assertEqual(exhausted.observe(None), 'retry')
        self.assertEqual(exhausted.observe(('circle', 'red')), 'exhausted')

    def test_terminal_states_reject_observations_without_changing_counts(self):
        for key in (None, ('circle', 'red')):
            gate = ScanConsensus(max_attempts=1, required_matches=1)
            self.assertEqual(gate.observe(key), 'exhausted' if key is None else 'accept')
            before = (gate.attempts, gate.matches, gate.candidate)
            with self.assertRaises(ValueError):
                gate.observe(('circle', 'red'))
            self.assertEqual((gate.attempts, gate.matches, gate.candidate), before)

    def test_invalid_keys_do_not_consume_budget_or_reset_streak(self):
        gate = ScanConsensus()
        gate.observe(('circle', 'red'))
        invalid = ['', 'circle', [], ['circle', 'red'], (), ('circle',),
                   ('circle', 'red', 'extra'), ('', 'red'), ('circle', ''),
                   ('circle', 1), (True, 'red'), ('   ', 'red')]
        for key in invalid:
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    gate.observe(key)
                self.assertEqual((gate.attempts, gate.matches, gate.candidate), (1, 1, ('circle', 'red')))
        self.assertEqual(gate.observe(('circle', 'red')), 'accept')

    def test_invalid_limits_are_rejected(self):
        for attempts, matches in [(0, 1), (21, 2), (2, 0), (2, 3), (-1, 1),
                                  (True, 1), (2, True), (2.0, 1), (2, 1.0), ('2', 1), (2, None)]:
            with self.subTest(attempts=attempts, matches=matches):
                with self.assertRaises(ValueError):
                    ScanConsensus(max_attempts=attempts, required_matches=matches)

    def test_inclusive_limit_boundaries(self):
        gate = ScanConsensus(max_attempts=1, required_matches=1)
        self.assertEqual(gate.observe(('circle', 'red')), 'accept')
        gate = ScanConsensus(max_attempts=20, required_matches=20)
        for _ in range(19):
            self.assertEqual(gate.observe(('circle', 'red')), 'retry')
        self.assertEqual(gate.observe(('circle', 'red')), 'accept')
        self.assertEqual((gate.attempts, gate.matches), (20, 20))


if __name__ == '__main__':
    unittest.main()
