"""Independent scene evidence against an explicitly recorded empty reference.

Compare spatial structure after removing smooth illumination differences. The
original colour frames and the manually recorded background remain unchanged;
this is independent of detector outputs and never learns an empty scene implicitly.
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
_CLEANUP_KERNEL = np.ones((7, 7), np.uint8)


def _components(mask, clean=False):
    mask = mask.astype(np.uint8)
    if clean:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _CLEANUP_KERNEL)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return float(mask.mean()), int(stats[1:, cv2.CC_STAT_AREA].max()) if count > 1 else 0


def _local_contrast(gray):
    mean = cv2.GaussianBlur(gray, (0, 0), 4)
    variance = cv2.GaussianBlur(gray * gray, (0, 0), 4) - mean * mean
    return np.sqrt(np.maximum(variance, 0))


def _assess_view(current, reference):
    current = cv2.GaussianBlur(current.astype(np.float32), (5, 5), 0)
    reference = cv2.GaussianBlur(reference.astype(np.float32), (5, 5), 0)
    delta = current - reference
    raw_difference = np.max(np.abs(delta), axis=2)
    # Remove only smooth shading/colour offsets. Object boundaries remain in
    # the residual even when an object's uniform interior follows this field.
    illumination = cv2.GaussianBlur(delta, (0, 0), 8)
    residual = np.max(np.abs(delta - illumination), axis=2)
    ratio, largest = _components(residual > 14, clean=True)
    weak_ratio, weak_largest = _components(residual > 10, clean=True)
    _, sharp_largest = _components(residual > 65)
    gray_current = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    gray_reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    reference_contrast = _local_contrast(gray_reference)
    current_contrast = _local_contrast(gray_current)
    # A low-contrast object can hide background texture without a strong edge.
    # Require a coherent region, rather than individual disappearing noise dots.
    texture_ratio, texture_largest = _components(
        (reference_contrast > 4) & (current_contrast < .35 * reference_contrast), clean=True)
    lighting_shift = float(np.max(np.abs(np.median(delta[::4, ::4], axis=(0, 1)))))
    clipped_ratio = float(((gray_current <= 2) | (gray_current >= 253)).mean())
    scores = {'method': 'illumination_structure_v2',
              'changed_ratio': ratio, 'largest_component': largest,
              'mean_difference': float(raw_difference.mean()),
              'residual_mean': float(residual.mean()),
              'weak_changed_ratio': weak_ratio, 'weak_largest_component': weak_largest,
              'texture_loss_ratio': texture_ratio, 'texture_loss_component': texture_largest,
              'sharp_component': sharp_largest, 'lighting_shift': lighting_shift,
              'clipped_ratio': clipped_ratio}
    # Extreme exposure/clipping or broad structural disagreement still needs
    # inspection. Never adapt the stored reference to an unconfirmed scene.
    if lighting_shift > 55 or clipped_ratio > .20 or ratio > .25:
        return 'unknown', 'exposure or background structure changed too much', scores
    # At 640x480, a 7x7 opening removes narrow glints/noise; a surviving region
    # of 96 pixels is occupied evidence. Test physical objects and field frames
    # when changing these spatial thresholds, not just detector confidence.
    if largest >= 96 or texture_largest >= 1600:
        return 'occupied', 'coherent object boundary or background texture occlusion', scores
    if weak_largest >= 480 or sharp_largest >= 4:
        return 'unknown', 'localized scene change needs confirmation', scores
    return 'empty', 'background structure matches after illumination compensation', scores


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
            state, reason, score = _assess_view(current, reference)
            raw_peak = int(np.abs(current.astype(np.int16) - reference.astype(np.int16)).max())
            scores[role] = dict(score, raw_peak=raw_peak, state=state, reason=reason)
            states.append(state)
        if 'occupied' in states:
            return result('occupied', 'local scene change in at least one view', scores)
        if 'unknown' in states:
            return result('unknown', 'exposure, structure or localized change needs confirmation', scores)
        return result('empty', 'both views match the empty scene after illumination compensation', scores)
