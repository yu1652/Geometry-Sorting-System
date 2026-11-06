"""Independent scene evidence against an explicitly recorded empty reference.

Thresholds are conservative starting points for on-site acceptance testing, not
proof that every object is perceptible. This module never consults detector
outputs and never learns an empty scene implicitly.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
from typing import Optional
import uuid
from zipfile import BadZipFile

import cv2
import numpy as np


@dataclass(frozen=True)
class PresenceResult:
    state: str
    reason: str
    scores: dict
    reference_bytes: Optional[bytes] = None
    reference_sha256: Optional[str] = None


_ROLES = ('left', 'right')
_MAX_REFERENCE_BYTES = 5_000_000


def _sources(sources):
    if not isinstance(sources, Mapping):
        raise ValueError('left/right sources are required')
    result = {}
    for role in _ROLES:
        source = sources.get(role)
        if isinstance(source, bool) or not isinstance(source, (str, int)) or not str(source).strip():
            raise ValueError('invalid source for ' + role)
        result[role] = str(source)
    return result


def _frames(frames):
    if not isinstance(frames, Mapping):
        raise ValueError('left/right frames are required')
    result = {}
    for role in _ROLES:
        frame = frames.get(role)
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.shape != (480, 640, 3):
            raise ValueError('invalid 640x480 BGR frame for ' + role)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.median(gray))
        if brightness <= 12 or brightness >= 243 or float(gray.std()) < 3:
            raise ValueError('dark, bright or textureless frame for ' + role)
        result[role] = frame.copy()
    return result


class EmptySceneReference:
    """Persist a manually confirmed empty dual-view scene and assess new frames."""

    def __init__(self, path):
        self.path = Path(path)

    def record(self, frames, sources):
        """Save only explicitly supplied empty-area captures; preserve old on error."""
        images = _frames(frames)
        bindings = _sources(sources)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + '.' + uuid.uuid4().hex + '.tmp')
        try:
            # Passing a file handle prevents NumPy silently adding a .npz suffix.
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, left=images['left'], right=images['right'],
                                    metadata=np.array(json.dumps({'version': 1, 'sources': bindings})))
            os.replace(temporary, self.path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def assess(self, frames, sources):
        """Return empty/occupied/unknown without treating YOLO absence as evidence."""
        reference_bytes = None
        reference_sha256 = None

        def result(state, reason, scores=None):
            return PresenceResult(state, reason, scores if scores is not None else {},
                                  reference_bytes, reference_sha256)

        try:
            images = _frames(frames)
            bindings = _sources(sources)
            if self.path.stat().st_size > _MAX_REFERENCE_BYTES:
                return result('unknown', 'reference exceeds 5 MB size limit')
            snapshot = self.path.read_bytes()
            if len(snapshot) > _MAX_REFERENCE_BYTES:
                return result('unknown', 'reference exceeds 5 MB size limit')
            # Decode and identify the same immutable snapshot. Reopening the
            # path could race with an operator recording a replacement scene.
            reference_bytes = snapshot
            reference_sha256 = sha256(snapshot).hexdigest()
            with np.load(BytesIO(snapshot), allow_pickle=False) as reference:
                metadata = json.loads(str(reference['metadata'].item()))
                if not isinstance(metadata, dict) or metadata.get('version') != 1:
                    return result('unknown', 'invalid reference metadata')
                if metadata.get('sources') != bindings:
                    return result('unknown', 'camera sources differ from recorded reference')
                backgrounds = _frames({role: reference[role] for role in _ROLES})
        except (OSError, ValueError, TypeError, KeyError, EOFError, BadZipFile) as exc:
            return result('unknown', 'reference or frames unavailable: ' + str(exc))

        scores = {}
        states = []
        for role in _ROLES:
            current = images[role]
            reference = backgrounds[role]
            # Blur suppresses ordinary sensor noise. Channelwise differences
            # retain colour changes even for objects with similar grey levels.
            blurred_current = cv2.GaussianBlur(current.astype(np.float32), (5, 5), 0)
            blurred_reference = cv2.GaussianBlur(reference.astype(np.float32), (5, 5), 0)
            difference = np.max(np.abs(blurred_current - blurred_reference), axis=2)
            changed = (difference > 14).astype(np.uint8)
            ratio = float(changed.mean())
            mean_difference = float(difference.mean())
            count, _, stats, _ = cv2.connectedComponentsWithStats(changed, connectivity=8)
            largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if count > 1 else 0
            # A strong threshold alone misses a low-contrast object that covers
            # texture without changing average brightness. Coherent differences
            # above the tested +/-2 sensor noise range make emptiness uncertain.
            weak_changed = (difference > 3).astype(np.uint8)
            weak_ratio = float(weak_changed.mean())
            weak_count, _, weak_stats, _ = cv2.connectedComponentsWithStats(weak_changed, connectivity=8)
            weak_largest = int(weak_stats[1:, cv2.CC_STAT_AREA].max()) if weak_count > 1 else 0
            raw_peak = int(np.abs(current.astype(np.int16) - reference.astype(np.int16)).max())
            scores[role] = {'changed_ratio': ratio, 'largest_component': largest,
                            'mean_difference': mean_difference, 'raw_peak': raw_peak,
                            'weak_changed_ratio': weak_ratio, 'weak_largest_component': weak_largest}
            # Widespread differences indicate changed lighting/camera position
            # rather than dependable evidence of an object or an empty scene.
            if weak_ratio > .30 or ratio > .30 or mean_difference > 25:
                states.append('unknown')
            elif largest >= 12 or ratio > .00015:
                states.append('occupied')
            # Keep sparse high-contrast changes ambiguous instead of confidently
            # discarding a tiny object as sensor noise. Both area tests and this
            # raw guard are needed: Gaussian smoothing can hide a single pixel.
            elif weak_largest >= 12 or largest > 0 or raw_peak > 35:
                states.append('unknown')
            else:
                states.append('empty')
        if 'occupied' in states:
            return result('occupied', 'local scene change in at least one view', scores)
        if 'unknown' in states:
            return result('unknown', 'global, weak or sparse ambiguous scene change', scores)
        return result('empty', 'both views match the recorded empty scene', scores)
