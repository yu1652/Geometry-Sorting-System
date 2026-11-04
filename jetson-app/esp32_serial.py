"""Line-oriented ESP32 serial transport, independent of the serial backend."""
from pathlib import Path

BAUD_RATE = 115200


class SerialTransportError(RuntimeError):
    """A serial read or write failed."""


class SerialLineTransport:
    def __init__(self, port, max_line_bytes=256):
        if max_line_bytes < 1:
            raise ValueError("max_line_bytes must be positive")
        self.port = port
        self.max_line_bytes = max_line_bytes
        self._buffer = bytearray()
        self._discard = False

    def read_lines(self):
        try:
            data = self.port.read(4096)
        except Exception as exc:
            raise SerialTransportError("serial read failed: {}".format(exc)) from exc
        lines = []
        for byte in data:
            if byte == 10:
                if not self._discard:
                    frame = self._buffer
                    if frame.endswith(b"\r"):
                        frame = frame[:-1]
                    if frame and len(frame) <= self.max_line_bytes and all(32 <= b <= 126 for b in frame):
                        lines.append(frame.decode("ascii"))
                self._buffer = bytearray()
                self._discard = False
            elif not self._discard:
                self._buffer.append(byte)
                # One extra byte is allowed solely for the optional CR delimiter.
                if len(self._buffer) > self.max_line_bytes + 1 or (len(self._buffer) == self.max_line_bytes + 1 and byte != 13):
                    self._buffer.clear()
                    self._discard = True
        return lines

    def write_line(self, line):
        if not isinstance(line, str) or not line or len(line) > self.max_line_bytes or any(not 32 <= ord(c) <= 126 for c in line):
            raise ValueError("line must contain printable ASCII within the size limit")
        frame = (line + "\n").encode("ascii")
        offset = 0
        while offset < len(frame):
            try:
                count = self.port.write(frame[offset:])
            except Exception as exc:
                raise SerialTransportError("serial write failed: {}".format(exc)) from exc
            if not isinstance(count, int) or count <= 0 or count > len(frame) - offset:
                raise SerialTransportError("serial write made no valid progress")
            offset += count


def list_stable_serial_paths(directory="/dev/serial/by-id"):
    path = Path(directory)
    if not path.is_dir():
        return []
    return sorted(str(entry) for entry in path.iterdir() if entry.exists())
