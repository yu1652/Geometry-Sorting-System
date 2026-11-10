"""Durable first-appearance bin assignments, independent of UI and hardware.

Reserve before sending SORT. Complete only after the matching MCU DONE.
An unresolved reservation intentionally survives STOP and process restarts.
"""

import copy
import json
import os
from pathlib import Path
import re
import tempfile
import uuid


class CapacityFull(ValueError):
    """All six bins are already assigned to other categories."""


def _sync_directory(directory):
    """Persist directory entries on the Linux target before motion is authorized."""
    if os.name == 'nt':
        return
    descriptor = os.open(str(directory), os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class AutoBinSession:
    def __init__(self, path, class_names, colors, *, load_existing=True):
        self.path = Path(path)
        self._shapes = frozenset(class_names)
        self._colors = frozenset(colors)
        if not all(isinstance(value, str) and value for value in self._shapes | self._colors):
            raise ValueError('Category vocabularies must contain nonempty strings')
        self._state = self._empty_state()
        if load_existing and self.path.exists():
            try:
                with self.path.open(encoding='utf-8') as stream:
                    state = json.load(stream)
            except (ValueError, UnicodeError) as exc:
                raise ValueError('Invalid saved automatic bin session') from exc
            self._validate(state)
            self._state = state

    @staticmethod
    def _empty_state():
        return {'version': 1, 'round_id': uuid.uuid4().hex, 'entries': [], 'pending': None}

    @property
    def bindings(self):
        return {(entry['shape'], entry['color']): entry['bin']
                for entry in self._state['entries']}

    @property
    def counts(self):
        result = [0] * 6
        for entry in self._state['entries']:
            result[entry['bin'] - 1] = entry['count']
        return tuple(result)

    @property
    def total(self):
        return sum(self.counts)

    @property
    def pending(self):
        return copy.deepcopy(self._state['pending'])

    @property
    def round_id(self):
        return self._state['round_id']

    @property
    def entries(self):
        return copy.deepcopy(self._state['entries'])

    def _validate_category(self, shape, color):
        if (not isinstance(shape, str) or not isinstance(color, str)
                or shape not in self._shapes or color not in self._colors):
            raise ValueError('Unknown shape/color category')

    def _validate(self, state):
        if (not isinstance(state, dict)
                or set(state) != {'version', 'round_id', 'entries', 'pending'}
                or type(state['version']) is not int or state['version'] != 1):
            raise ValueError('Invalid automatic bin session schema/version')
        if (not isinstance(state['round_id'], str)
                or re.fullmatch(r'[A-Za-z0-9_-]{1,128}', state['round_id']) is None):
            raise ValueError('Invalid round identifier')
        entries = state['entries']
        if not isinstance(entries, list) or len(entries) > 6:
            raise ValueError('Invalid session entries')
        categories = set()
        for bin_id, entry in enumerate(entries, 1):
            if not isinstance(entry, dict) or set(entry) != {'shape', 'color', 'bin', 'count'}:
                raise ValueError('Invalid session entry')
            self._validate_category(entry['shape'], entry['color'])
            category = (entry['shape'], entry['color'])
            if category in categories:
                raise ValueError('Duplicate category')
            categories.add(category)
            if type(entry['bin']) is not int or entry['bin'] != bin_id:
                raise ValueError('Bins must be unique and contiguous in assignment order')
            if type(entry['count']) is not int or entry['count'] < 0:
                raise ValueError('Counts must be nonnegative integers')
        pending = state['pending']
        if pending is not None:
            if not isinstance(pending, dict) or set(pending) != {'shape', 'color', 'bin'}:
                raise ValueError('Invalid pending reservation')
            self._validate_category(pending['shape'], pending['color'])
            if type(pending['bin']) is not int or not any(
                    all(entry[key] == pending[key] for key in ('shape', 'color', 'bin'))
                    for entry in entries):
                raise ValueError('Pending reservation must reference its assigned entry')

    def _persist(self, state):
        """Publish memory only after a flushed, atomic replacement succeeds."""
        temporary = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # A prior failed sync may leave new directories visible but not
            # durable. Sync the ancestor chain on retries as well as creation.
            directories = (self.path.parent,) + tuple(self.path.parent.parents)
            for directory in reversed(directories):
                _sync_directory(directory)
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                             dir=str(self.path.parent),
                                             prefix=self.path.name + '.', suffix='.tmp',
                                             delete=False) as stream:
                temporary = stream.name
                json.dump(state, stream, ensure_ascii=False, allow_nan=False, indent=2)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            temporary = None
            _sync_directory(self.path.parent)
        finally:
            if temporary is not None:
                os.unlink(temporary)
        self._state = state

    def reserve(self, shape, color):
        self._validate_category(shape, color)
        if self._state['pending'] is not None:
            raise ValueError('Resolve the pending reservation before sorting again')
        state = copy.deepcopy(self._state)
        bin_id = self.bindings.get((shape, color))
        if bin_id is None:
            if len(state['entries']) == 6:
                raise CapacityFull('All six bins are assigned')
            bin_id = len(state['entries']) + 1
            state['entries'].append({'shape': shape, 'color': color, 'bin': bin_id, 'count': 0})
        state['pending'] = {'shape': shape, 'color': color, 'bin': bin_id}
        self._persist(state)
        return bin_id

    def complete(self, bin_id):
        pending = self._state['pending']
        if type(bin_id) is not int or pending is None or pending['bin'] != bin_id:
            raise ValueError('Completion does not match the pending reservation')
        state = copy.deepcopy(self._state)
        state['entries'][bin_id - 1]['count'] += 1
        state['pending'] = None
        self._persist(state)

    def cancel_unsent(self, bin_id, *, new_binding):
        """Undo a reservation only after the caller knows SORT was never sent."""
        pending = self._state['pending']
        if type(bin_id) is not int or pending is None or pending['bin'] != bin_id:
            raise ValueError('Cancellation does not match the pending reservation')
        state = copy.deepcopy(self._state)
        if new_binding:
            entry = state['entries'][-1]
            if entry['bin'] != bin_id or entry['count'] != 0:
                raise ValueError('Only the newest unused bin can be released')
            state['entries'].pop()
        state['pending'] = None
        self._persist(state)

    def clear(self):
        self._persist(self._empty_state())
