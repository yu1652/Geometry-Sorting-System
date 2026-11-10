import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from auto_bins import AutoBinSession, CapacityFull


class AutoBinSessionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'round.json'
        self.shapes = ['circle', 'square', 'triangle', 'hexagon']
        self.colors = ['red', 'blue']
        self.session = self.reload()

    def reload(self):
        return AutoBinSession(self.path, self.shapes, self.colors)

    def finish(self, shape, color):
        bin_id = self.session.reserve(shape, color)
        self.session.complete(bin_id)
        return bin_id

    def test_missing_file_is_empty_without_writing(self):
        self.assertFalse(self.path.exists())
        self.assertEqual(self.session.bindings, {})
        self.assertEqual(self.session.counts, (0, 0, 0, 0, 0, 0))
        self.assertEqual(self.session.total, 0)
        self.assertIsNone(self.session.pending)
        self.assertTrue(self.session.round_id)

    def test_first_appearance_orders_bins_and_color_distinguishes_categories(self):
        self.assertEqual(self.finish('square', 'blue'), 1)
        self.assertEqual(self.finish('circle', 'red'), 2)
        self.assertEqual(self.finish('square', 'red'), 3)
        self.assertEqual(self.finish('square', 'blue'), 1)
        self.assertEqual(self.session.bindings, {
            ('square', 'blue'): 1, ('circle', 'red'): 2, ('square', 'red'): 3})
        self.assertEqual(self.session.counts, (2, 1, 1, 0, 0, 0))
        self.assertEqual(self.session.total, 4)
        self.assertEqual(self.reload().entries, self.session.entries)

    def test_seventh_category_is_rejected_but_existing_category_still_works(self):
        for shape in self.shapes[:3]:
            for color in self.colors:
                self.finish(shape, color)
        before = self.path.read_bytes()
        with self.assertRaises(CapacityFull):
            self.session.reserve('hexagon', 'red')
        self.assertEqual(self.path.read_bytes(), before)
        self.assertIsNone(self.session.pending)
        self.assertEqual(self.finish('circle', 'red'), 1)

    def test_unsent_reservation_cancellation_keeps_counts_and_releases_new_bin(self):
        first = self.finish('circle', 'red')
        self.assertEqual(self.session.reserve('circle', 'red'), first)
        self.session.cancel_unsent(first, new_binding=False)
        second = self.session.reserve('square', 'blue')
        self.session.cancel_unsent(second, new_binding=True)
        recovered = self.reload()
        self.assertIsNone(recovered.pending)
        self.assertEqual(recovered.bindings, {('circle', 'red'): 1})
        self.assertEqual(recovered.total, 1)
        self.assertEqual(recovered.reserve('triangle', 'red'), 2)

    def test_reservation_is_durable_pending_and_only_matching_complete_counts(self):
        self.assertEqual(self.session.reserve('circle', 'red'), 1)
        recovered = self.reload()
        self.assertEqual(recovered.pending, {'shape': 'circle', 'color': 'red', 'bin': 1})
        self.assertEqual(recovered.bindings, {('circle', 'red'): 1})
        self.assertEqual(recovered.counts, (0, 0, 0, 0, 0, 0))
        before = self.path.read_bytes()
        for action in (lambda: recovered.reserve('square', 'blue'),
                       lambda: recovered.complete(2), lambda: recovered.complete(True)):
            with self.assertRaises(ValueError):
                action()
            self.assertEqual(self.path.read_bytes(), before)
        recovered.complete(1)
        self.assertIsNone(recovered.pending)
        self.assertEqual(self.reload().counts, (1, 0, 0, 0, 0, 0))
        with self.assertRaises(ValueError):
            recovered.complete(1)

    def test_clear_starts_new_round_and_removes_unresolved_pending(self):
        self.finish('circle', 'red')
        self.session.reserve('square', 'blue')
        old_round = self.session.round_id
        self.session.clear()
        recovered = self.reload()
        self.assertNotEqual(recovered.round_id, old_round)
        self.assertEqual(recovered.entries, [])
        self.assertEqual(recovered.total, 0)
        self.assertIsNone(recovered.pending)
        self.assertEqual(recovered.reserve('triangle', 'blue'), 1)

    def test_returned_collections_cannot_mutate_session(self):
        self.session.reserve('circle', 'red')
        self.session.entries[0]['count'] = 100
        self.session.bindings[('square', 'red')] = 2
        self.session.pending['bin'] = 6
        self.assertEqual(self.session.total, 0)
        self.assertEqual(self.session.pending['bin'], 1)
        self.assertEqual(len(self.session.bindings), 1)

    def test_unknown_or_nonstring_categories_leave_state_untouched(self):
        for shape, color in [('unknown', 'red'), ('circle', 'green'), (True, 'red'),
                             ([], 'red'), ('circle', None)]:
            with self.subTest(shape=shape, color=color):
                with self.assertRaises(ValueError):
                    self.session.reserve(shape, color)
        self.assertFalse(self.path.exists())
        self.assertEqual(self.session.entries, [])

    def test_invalid_persisted_states_raise_without_overwriting(self):
        self.session.reserve('circle', 'red')
        valid = json.loads(self.path.read_text(encoding='utf-8'))
        invalid = [None, [], {}, dict(valid, version=999), dict(valid, version=True),
                   dict(valid, round_id=''), dict(valid, round_id=3),
                   dict(valid, pending={'shape': 'square', 'color': 'red', 'bin': 1})]
        for field, value in [('shape', 'unknown'), ('color', 'green'), ('bin', 2),
                             ('bin', True), ('count', -1), ('count', True),
                             ('count', 0.5), ('count', float('nan'))]:
            state = copy.deepcopy(valid)
            state['entries'][0][field] = value
            invalid.append(state)
        duplicate = copy.deepcopy(valid)
        duplicate['entries'].append(dict(duplicate['entries'][0], bin=2))
        invalid.append(duplicate)
        duplicate_bin = copy.deepcopy(valid)
        duplicate_bin['entries'].append(dict(duplicate_bin['entries'][0], shape='square'))
        invalid.append(duplicate_bin)
        for state in invalid:
            with self.subTest(state=state):
                content = json.dumps(state).encode('utf-8')
                self.path.write_bytes(content)
                with self.assertRaises(ValueError):
                    self.reload()
                self.assertEqual(self.path.read_bytes(), content)
        self.path.write_text('{broken', encoding='utf-8')
        with self.assertRaises(ValueError):
            self.reload()
        self.assertEqual(self.path.read_text(encoding='utf-8'), '{broken')

    def test_disk_failure_never_advances_memory_or_existing_file(self):
        self.finish('circle', 'red')
        before_bytes = self.path.read_bytes()
        before_round = self.session.round_id
        with patch('auto_bins.os.replace', side_effect=OSError('disk unavailable')):
            with self.assertRaises(OSError):
                self.session.reserve('square', 'blue')
            self.assertEqual(self.session.bindings, {('circle', 'red'): 1})
            self.assertIsNone(self.session.pending)
            with self.assertRaises(OSError):
                self.session.clear()
        self.assertEqual(self.session.round_id, before_round)
        self.assertEqual(self.path.read_bytes(), before_bytes)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_explicit_fresh_session_can_clear_corrupt_round(self):
        self.path.write_text('{broken', encoding='utf-8')
        with self.assertRaises(ValueError):
            self.reload()
        fresh = AutoBinSession(self.path, self.shapes, self.colors, load_existing=False)
        self.assertEqual(self.path.read_text(encoding='utf-8'), '{broken')
        fresh.clear()
        self.assertEqual(self.reload().entries, [])
        self.assertIsNone(self.reload().pending)

    def test_first_reservation_creates_missing_data_folder(self):
        path = self.path.parent / 'new' / 'nested' / 'round.json'
        session = AutoBinSession(path, self.shapes, self.colors)
        self.assertFalse(path.exists())
        session.reserve('circle', 'red')
        self.assertTrue(path.is_file())
        restored = AutoBinSession(path, self.shapes, self.colors)
        self.assertEqual(restored.bindings, {('circle','red'):1})

    def test_reservation_syncs_directory_after_publication_and_parent_creation(self):
        path = self.path.parent / 'new' / 'nested' / 'round.json'
        session = AutoBinSession(path, self.shapes, self.colors)
        events = []
        import os
        replace = os.replace
        def publish(source, destination):
            events.append(('publish', Path(destination)))
            replace(source, destination)
        with patch('auto_bins.os.replace', side_effect=publish), patch(
                'auto_bins._sync_directory', side_effect=lambda p: events.append(('sync', Path(p)))):
            session.reserve('circle','red')
        publication = events.index(('publish', path))
        self.assertIn(('sync', path.parent), events[publication + 1:])
        self.assertIn(('sync', self.path.parent), events[:publication])
        self.assertIn(('sync', path.parent.parent), events[:publication])

    def test_completion_disk_failure_retains_pending_and_count(self):
        self.finish('circle', 'red')
        self.session.reserve('circle', 'red')
        before_bytes = self.path.read_bytes()
        with patch('auto_bins.os.fsync', side_effect=OSError('disk unavailable')):
            with self.assertRaises(OSError):
                self.session.complete(1)
        self.assertEqual(self.session.counts, (1, 0, 0, 0, 0, 0))
        self.assertIsNotNone(self.session.pending)
        self.assertEqual(self.path.read_bytes(), before_bytes)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])


if __name__ == '__main__':
    unittest.main()
