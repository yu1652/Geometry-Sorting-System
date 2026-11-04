"""Explicit capture sources for the left, right and top camera roles."""
import os
from collections.abc import Mapping

import cv2
import numpy as np


CAMERA_ROLES = ('left', 'right', 'top')


class CameraError(RuntimeError):
    """A camera configuration or lifecycle operation failed."""


class CameraSource:
    """Own one OpenCV-compatible capture for a device path or recording."""

    def __init__(self, source, capture_factory=None):
        try:
            path = os.fspath(source)
        except TypeError as exc:
            raise CameraError('Expected a stable device path or recording path') from exc
        if not isinstance(path, str) or not path.strip() or '\x00' in path:
            raise CameraError('Expected a non-empty text path')
        if path.startswith('/dev/'):
            parts = path.split('/')
            if (
                len(parts) != 5
                or parts[:3] != ['', 'dev', 'v4l']
                or parts[3] not in ('by-id', 'by-path')
                or parts[4] in ('', '.', '..')
            ):
                raise CameraError('Expected a stable /dev/v4l path, got {!r}'.format(path))
        self.source = path
        self._factory = cv2.VideoCapture if capture_factory is None else capture_factory
        if not callable(self._factory):
            raise CameraError('Capture factory must be callable')
        self._capture = None

    def open(self):
        """Open this capture if it is currently closed."""
        if self._capture is not None:
            return
        capture = None
        try:
            capture = self._factory(self.source)
            if not capture.isOpened():
                raise CameraError('Capture backend could not open the source')
        except Exception as exc:
            message = 'Could not open {!r}: {}'.format(self.source, exc)
            if capture is not None:
                try:
                    capture.release()
                except Exception as release_error:
                    message += ' (release failed: {})'.format(release_error)
            raise CameraError(message) from exc
        self._capture = capture

    def read(self):
        """Return an owned, contiguous BGR frame, or None when unavailable."""
        if self._capture is None:
            return None
        try:
            ok, frame = self._capture.read()
        except Exception:
            return None
        if not ok or not isinstance(frame, np.ndarray):
            return None
        if frame.ndim != 3 or frame.shape[2] != 3 or not frame.size:
            return None
        return frame.copy(order='C')

    def close(self):
        """Release this capture and clear its state even if release fails."""
        capture = self._capture
        self._capture = None
        if capture is not None:
            try:
                capture.release()
            except Exception as exc:
                raise CameraError('Could not close {!r}: {}'.format(self.source, exc)) from exc


class CameraService:
    """Manage explicitly configured captures without opening them on creation."""

    def __init__(self, sources, capture_factory=None):
        if not isinstance(sources, Mapping):
            raise CameraError('Camera sources must map roles to paths')
        self._sources = {}
        for role, source in sources.items():
            if role not in CAMERA_ROLES:
                raise CameraError('Unknown camera role {!r}'.format(role))
            try:
                self._sources[role] = CameraSource(source, capture_factory)
            except CameraError as exc:
                raise CameraError('Camera {!r}: {}'.format(role, exc)) from exc

    def _source_for(self, role):
        if role not in CAMERA_ROLES or role not in self._sources:
            raise CameraError('Camera role {!r} is not configured'.format(role))
        return self._sources[role]

    def open(self, role):
        source = self._source_for(role)
        try:
            source.open()
        except CameraError as exc:
            raise CameraError('Camera {!r}: {}'.format(role, exc)) from exc

    def read(self, role):
        return self._source_for(role).read()

    def close(self, role=None):
        """Close one role, or attempt to close every configured role."""
        roles = self._sources if role is None else (role,)
        errors = []
        for camera_role in roles:
            source = self._source_for(camera_role)
            try:
                source.close()
            except CameraError as exc:
                errors.append('Camera {!r}: {}'.format(camera_role, exc))
        if errors:
            raise CameraError(' | '.join(errors))
