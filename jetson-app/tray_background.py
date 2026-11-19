"""Optional raw tray reference storage, independent of motion and cameras."""
from io import BytesIO
import os
from pathlib import Path
import stat
import struct
import tempfile
from zipfile import BadZipFile, ZipFile, ZIP_DEFLATED, ZIP_STORED
import zlib

import numpy as np


SCHEMA_VERSION = 1
FRAME_SHAPE = (480, 640, 3)
MAX_SOURCE_LENGTH = 4096
MAX_ARCHIVE_BYTES = 2_000_000
MAX_UNCOMPRESSED_BYTES = 2_000_000
MAX_HEADER_BYTES = 4096
_MEMBERS = {'version.npy', 'frame.npy', 'source_top.npy', 'roi.npy'}


def _source(value):
    if (not isinstance(value, str) or not value.strip() or '\x00' in value
            or len(value) > MAX_SOURCE_LENGTH):
        raise ValueError('Invalid tray background camera identity')
    return value


def _roi(value):
    try:
        values = tuple(value)
    except TypeError as error:
        raise ValueError('Tray background ROI requires four integers') from error
    if (len(values) != 4 or any(isinstance(v, (bool, np.bool_)) or
            not isinstance(v, (int, np.integer)) for v in values)):
        raise ValueError('Tray background ROI requires four integers')
    x, y, width, height = (int(v) for v in values)
    if (x < 0 or y < 0 or width <= 0 or height <= 0
            or x + width > 640 or y + height > 480):
        raise ValueError('Tray background ROI is outside the full image')
    return x, y, width, height


def _read_array(raw, name):
    """Validate a bounded NPY header before constructing any shaped array.

    Reading only these primitive dtypes never invokes pickle. Checking the
    declared shape first also prevents tiny forged archives allocating huge
    arrays through a malicious NPY header.
    """
    stream = BytesIO(raw)
    version = np.lib.format.read_magic(stream)
    if version not in ((1, 0), (2, 0)):
        raise ValueError('Unsupported tray background NPY version')
    length_size = 2 if version == (1, 0) else 4
    length = stream.read(length_size)
    if len(length) != length_size:
        raise ValueError('Truncated tray background array header')
    header_size = struct.unpack('<H' if length_size == 2 else '<I', length)[0]
    if header_size > MAX_HEADER_BYTES or stream.tell() + header_size > len(raw):
        raise ValueError('Invalid tray background array header size')
    stream.seek(8)
    reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
              else np.lib.format.read_array_header_2_0)
    shape, fortran, dtype = reader(stream)
    if dtype.hasobject or fortran:
        raise ValueError('Tray background arrays must use plain C-order values')
    if name == 'frame.npy':
        valid = shape == FRAME_SHAPE and dtype == np.dtype(np.uint8)
        count = 480 * 640 * 3
    elif name == 'roi.npy':
        valid = shape == (4,) and dtype == np.dtype(np.int64)
        count = 4
    elif name == 'version.npy':
        valid = shape == () and dtype == np.dtype(np.int64)
        count = 1
    else:
        valid = (shape == () and dtype.kind == 'U' and
                 0 < dtype.itemsize <= MAX_SOURCE_LENGTH * 4)
        count = 1
    if not valid or len(raw) - stream.tell() != count * dtype.itemsize:
        raise ValueError('Invalid tray background array shape, dtype or data size: ' + name)
    return np.frombuffer(raw, dtype=dtype, count=count, offset=stream.tell()).reshape(shape).copy()


class TrayBackgroundReference:
    """A copied full raw BGR frame bound to a top camera and pixel ROI.

    This object neither judges scene emptiness nor participates in startup.
    Mechanical calibration points are intentionally not part of its identity.
    """

    def __init__(self, frame, source_top, roi):
        if (not isinstance(frame, np.ndarray) or frame.shape != FRAME_SHAPE
                or frame.dtype != np.dtype(np.uint8)):
            raise ValueError('Tray background requires a raw uint8 480x640 BGR frame')
        self._source_top = _source(source_top)
        self._roi = _roi(roi)
        self._frame = frame.copy()

    @property
    def frame(self):
        return self._frame.copy()

    @property
    def source_top(self):
        return self._source_top

    @property
    def roi(self):
        return self._roi

    def save(self, path):
        """Replace only after a complete flush; failures retain the old file."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(prefix='.tray-background-', suffix='.npz',
                    dir=str(destination.parent), delete=False) as stream:
                temporary = Path(stream.name)
                np.savez_compressed(stream, version=np.array(SCHEMA_VERSION, np.int64),
                    frame=self._frame, source_top=np.array(self._source_top),
                    roi=np.array(self._roi, np.int64))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(str(temporary), str(destination))
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    @classmethod
    def load(cls, path, expected_source_top=None, expected_roi=None):
        """Load only this bounded primitive schema; reject camera/crop changes."""
        if expected_source_top is not None:
            expected_source_top = _source(expected_source_top)
        if expected_roi is not None:
            expected_roi = _roi(expected_roi)
        try:
            with Path(path).open('rb') as stream:
                stream.seek(0, os.SEEK_END)
                if stream.tell() > MAX_ARCHIVE_BYTES:
                    raise ValueError('Tray background archive is too large')
                stream.seek(0)
                with ZipFile(stream) as archive:
                    entries = archive.infolist()
                    names = [entry.filename for entry in entries]
                    if len(names) != len(set(names)) or set(names) != _MEMBERS:
                        raise ValueError('Unexpected or duplicate tray background archive members')
                    if (sum(entry.file_size for entry in entries) > MAX_UNCOMPRESSED_BYTES
                            or any(entry.is_dir() or entry.flag_bits & 1 or
                                   stat.S_ISLNK(entry.external_attr >> 16) or
                                   entry.compress_type not in (ZIP_STORED, ZIP_DEFLATED)
                                   for entry in entries)):
                        raise ValueError('Unsafe tray background archive contents')
                    values = {name: _read_array(archive.read(name), name) for name in names}
            if values['version.npy'].item() != SCHEMA_VERSION:
                raise ValueError('Unsupported tray background schema version')
            reference = cls(values['frame.npy'], values['source_top.npy'].item(),
                            values['roi.npy'])
            if expected_source_top is not None and reference.source_top != expected_source_top:
                raise ValueError('Tray background camera does not match the current top camera')
            if expected_roi is not None and reference.roi != expected_roi:
                raise ValueError('Tray background ROI does not match the current image crop')
            return reference
        except (BadZipFile, EOFError, KeyError, TypeError, UnicodeError, struct.error,
                zlib.error, NotImplementedError) as error:
            raise ValueError('Invalid tray background archive: ' + str(error)) from error
