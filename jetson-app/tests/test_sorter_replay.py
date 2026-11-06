import unittest
from types import SimpleNamespace

from sorter_simulation import SimulatedSorter


class RecordedSorterTests(unittest.TestCase):
    def setUp(self):
        from sorter_replay import RecordedSorter

        self.machine = SimulatedSorter()
        self.machine.confirm_origin()
        self.machine.start()
        self.frames = {"left": "L", "right": "R", "top": "T"}
        self.calls = []
        self.result = SimpleNamespace(
            detection=SimpleNamespace(class_name="yuan_huan_zhu"),
            color=SimpleNamespace(name="青色", valid_pixels=120),
            error=None,
        )
        self.replay = RecordedSorter(
            self.machine, self.frames.get, self.locate, self.classify,
            {("yuan_huan_zhu", "青色"): 2},
        )

    def locate(self, frame):
        self.calls.append(("locate", frame))
        return (85.0, 42.0)

    def classify(self, left, right):
        self.assertEqual(self.machine.snapshot().state, "wait_scan")
        self.calls.append(("classify", left, right))
        return self.result

    def test_location_then_scan_then_sort_and_done(self):
        self.assertEqual(self.replay.step(), "done")
        self.assertEqual(self.calls, [("locate", "T"), ("classify", "L", "R")])
        state = self.machine.snapshot()
        self.assertEqual(state.bin_counts, (0, 1, 0, 0, 0, 0))
        self.assertIn("HOST PUSH,85.0,42.0,mm", state.log)
        self.assertIn("HOST SORT,2", state.log)
        self.assertFalse(self.machine.handle_mcu("DONE"))
        self.assertEqual(self.machine.snapshot().completed_count, 1)

    def test_unbound_class_stops_without_done_or_count(self):
        self.result.detection.class_name = "ba_leng_zhu"
        with self.assertRaisesRegex(ValueError, "binding"):
            self.replay.step()
        state = self.machine.snapshot()
        self.assertEqual(state.state, "stopped")
        self.assertEqual(state.completed_count, 0)
        self.assertNotIn("MCU DONE", state.log)

    def test_missing_frame_does_not_reuse_previous_pair(self):
        self.frames["right"] = None
        with self.assertRaisesRegex(ValueError, "right"):
            self.replay.step()
        self.assertEqual(self.calls, [])
        self.assertEqual(self.machine.snapshot().state, "stopped")

    def test_no_target_waits_without_scan(self):
        self.replay.locate = lambda frame: None
        self.assertEqual(self.replay.step(), "no_target")
        state = self.machine.snapshot()
        self.assertEqual(state.state, "running")
        self.assertIn("HOST PUSH,WAIT", state.log)
        self.assertNotIn("MCU SCAN", state.log)

    def test_replay_loads_offset_calibration_using_machine_travel(self):
        import json
        import tempfile
        from pathlib import Path
        from sorter_replay import _calibration
        rows = [('P%d' % (j * 3 + i + 1), 211 + i * 133, 92 + j * 133,
                 70 + i * 56, 12 + j * 56) for j in range(3) for i in range(3)]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'calibration.json'
            path.write_text(json.dumps({'mode': 'simulation', 'points': rows}), encoding='utf8')
            mapping = _calibration(path)
            self.assertAlmostEqual(mapping.map_pixel(477, 92)[0], 182)
            mapping.validate_physical(260, 230)
