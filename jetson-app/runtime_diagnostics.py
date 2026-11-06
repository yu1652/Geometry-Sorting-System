"""Local, bounded diagnostic archives without GUI or hardware dependencies.

Only explicitly supplied configuration files are eligible for exports. Original
images and arrays are encoded in memory and are never modified.
"""

from datetime import datetime, timezone
from io import BytesIO
import json
import os
from pathlib import Path
import re
import threading
import time
import uuid
from zipfile import ZipFile, ZIP_DEFLATED

import cv2
import numpy as np


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError('Unsupported diagnostic metadata: %s' % type(value).__name__)


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, indent=2, default=_json_default).encode('utf-8')


def _fields(value, names):
    if value is None:
        return None
    return {name: getattr(value, name, None) for name in names}


class RuntimeDiagnostics:
    """Keep short runtime logs and copyable ZIPs beneath ``data_dir``.

    Calls raise filesystem/encoding errors for the caller to report. Every file
    handle closes before a call returns, including on failures.
    """

    def __init__(self, data_dir, max_samples=20, log_max_bytes=2_000_000):
        if max_samples < 1 or log_max_bytes < 1:
            raise ValueError('diagnostic retention limits must be positive')
        self.data_dir = Path(data_dir)
        self.max_samples = max_samples
        self.log_max_bytes = log_max_bytes
        self._lock = threading.RLock()

    def log(self, message):
        """Append a timestamped UTF-8 event and rotate at most three backups."""
        stamp = datetime.now(timezone.utc).isoformat()
        prefix = '%s monotonic=%.6f ' % (stamp, time.monotonic())
        # Keep multiline messages on one log record, retaining their contents.
        message = str(message).replace('\r', '\\r').replace('\n', '\\n')
        record = (prefix + message + '\n').encode('utf-8')
        if len(record) > self.log_max_bytes:
            record = record[:max(0, self.log_max_bytes - 1)].decode('utf-8', errors='ignore').encode('utf-8') + b'\n'
        with self._lock:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            path = self.data_dir / 'live_runtime.log'
            if path.exists() and path.stat().st_size + len(record) > self.log_max_bytes:
                for index in range(3, 0, -1):
                    source = path if index == 1 else self.data_dir / ('live_runtime.log.%d' % (index - 1))
                    target = self.data_dir / ('live_runtime.log.%d' % index)
                    if source.exists():
                        os.replace(source, target)
            with path.open('ab') as stream:
                stream.write(record)

    def _archive(self, directory, prefix, limit, write):
        directory.mkdir(parents=True, exist_ok=True)
        name = '%s_%s_%s.zip' % (prefix, datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'), uuid.uuid4().hex)
        target = directory / name
        temporary = directory / (name + '.tmp')
        try:
            with ZipFile(temporary, 'w', compression=ZIP_DEFLATED) as bundle:
                write(bundle)
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
        archives = sorted(directory.glob(prefix + '_*.zip'), key=lambda path: path.name)
        for old in archives[:-limit]:
            old.unlink()
        return target

    @staticmethod
    def _image(bundle, name, value):
        if value is None:
            return
        success, encoded = cv2.imencode('.png', value)
        if not success:
            raise ValueError('Could not encode diagnostic image: ' + name)
        bundle.writestr(name, encoded.tobytes())

    def save_sample(self, result, context):
        """Record detector evidence; pair images retain their original semantics.

        ``diagnostics`` may contain pair_bgr, features_bgr, raw_output and a
        captures mapping. In particular a preprocessed pair is never described
        or reconstructed as a raw colour capture.
        """
        metadata = {
            'detection': _fields(getattr(result, 'detection', None), ('class_id', 'class_name', 'confidence', 'box')),
            'color': _fields(getattr(result, 'color', None), ('name', 'confidence', 'valid_pixels')),
            'context': context,
        }
        for name in ('mask_status', 'hole_counts', 'color_sources', 'elapsed_ms', 'error'):
            metadata[name] = getattr(result, name, None)
        diagnostics = getattr(result, 'diagnostics', None) or {}

        def write(bundle):
            bundle.writestr('metadata.json', _json_bytes(metadata))
            reference_bytes = diagnostics.get('empty_reference_bytes')
            if reference_bytes is not None:
                bundle.writestr('empty_reference.npz', reference_bytes)
            for name in ('pair_bgr', 'features_bgr'):
                self._image(bundle, name + '.png', diagnostics.get(name))
            self._image(bundle, 'preview_bgr.png', getattr(result, 'preview_bgr', None))
            raw_output = diagnostics.get('raw_output')
            if raw_output is not None:
                buffer = BytesIO()
                np.save(buffer, raw_output, allow_pickle=False)
                bundle.writestr('raw_output.npy', buffer.getvalue())
            for index, (role, frame) in enumerate((diagnostics.get('captures') or {}).items()):
                safe_role = re.sub(r'[^A-Za-z0-9_-]', '_', str(role))[:80] or 'capture'
                # Include an index only if normalization creates a duplicate.
                name = 'captures/' + safe_role + '.png'
                if name in bundle.namelist():
                    name = 'captures/%s_%d.png' % (safe_role, index)
                self._image(bundle, name, frame)

        with self._lock:
            return self._archive(self.data_dir / 'diagnostic_samples', 'sample', self.max_samples, write)

    @staticmethod
    def _safe_extra(path):
        allowed = {'.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.csv', '.txt'}
        sensitive = re.compile(r'(secret|password|credential|token|private|api[_.-]?key|\.env)', re.I)
        # This explicit local camera reference is part of scene diagnostics,
        # rather than a generic allowance for arbitrary NumPy/model archives.
        empty_scene = path.name == 'recognition_empty.npz' and path.is_file() and path.stat().st_size <= 5_000_000
        return (not path.is_symlink() and path.is_file()
                and (path.suffix.lower() in allowed or empty_scene) and not sensitive.search(path.name))

    def export(self, extra_files=(), metadata=None):
        """Export existing logs/samples plus explicit regular config files.

        Directories, symlinks, binary/model extensions and secret filenames are
        skipped. No directory supplied in ``extra_files`` is ever traversed.
        """
        def write(bundle):
            bundle.writestr('metadata.json', _json_bytes(metadata if metadata is not None else {}))
            for suffix in ('', '.1', '.2', '.3'):
                path = self.data_dir / ('live_runtime.log' + suffix)
                if path.is_file() and not path.is_symlink():
                    bundle.write(path, path.name)
            samples = sorted((self.data_dir / 'diagnostic_samples').glob('sample_*.zip'))[-self.max_samples:]
            for path in samples:
                if path.is_file() and not path.is_symlink():
                    bundle.write(path, 'diagnostic_samples/' + path.name)
            for index, file in enumerate(extra_files):
                path = Path(file)
                if self._safe_extra(path):
                    name = 'extra_files/' + path.name
                    if name in bundle.namelist():
                        name = 'extra_files/%d_%s' % (index, path.name)
                    bundle.write(path, name)

        with self._lock:
            return self._archive(self.data_dir / 'diagnostic_exports', 'export', 3, write)
