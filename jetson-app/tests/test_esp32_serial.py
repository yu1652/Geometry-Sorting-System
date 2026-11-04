import tempfile
import unittest
from pathlib import Path

from esp32_serial import BAUD_RATE, SerialLineTransport, SerialTransportError, list_stable_serial_paths


class FakePort:
    def __init__(self, *chunks, write_limit=None):
        self.chunks = list(chunks)
        self.write_limit = write_limit
        self.written = bytearray()

    def read(self, size):
        chunk = self.chunks.pop(0) if self.chunks else b""
        if isinstance(chunk, OSError):
            raise chunk
        return chunk

    def write(self, data):
        if isinstance(self.write_limit, OSError):
            raise self.write_limit
        count = len(data) if self.write_limit is None else min(len(data), self.write_limit)
        self.written.extend(data[:count])
        return count


class SerialTransportTests(unittest.TestCase):
    def test_reads_split_and_consecutive_frames_without_losing_timeout_state(self):
        transport = SerialLineTransport(FakePort(b"PIN", b"", b"G\r", b"\nOK\nN", b"EXT\n"))
        self.assertEqual(BAUD_RATE, 115200)
        self.assertEqual(transport.read_lines(), [])
        self.assertEqual(transport.read_lines(), [])
        self.assertEqual(transport.read_lines(), [])
        self.assertEqual(transport.read_lines(), ["PING", "OK"])
        self.assertEqual(transport.read_lines(), ["NEXT"])

    def test_discards_bad_frames_and_recovers_only_after_their_delimiters(self):
        transport = SerialLineTransport(
            FakePort(b"\n\xffbad\n\x00bad\nA\rB\n12345", b"6OK\n1234\r\nOK\n"),
            max_line_bytes=4,
        )
        self.assertEqual(transport.read_lines(), [])
        self.assertEqual(transport.read_lines(), ["1234", "OK"])

    def test_writes_complete_ascii_frame_across_partial_writes(self):
        port = FakePort(write_limit=2)
        SerialLineTransport(port).write_line("SET 1")
        self.assertEqual(port.written, b"SET 1\n")

    def test_rejects_invalid_frames_before_writing(self):
        port = FakePort()
        transport = SerialLineTransport(port, max_line_bytes=4)
        for line in ("", "A\nB", "A\r", "A\t", "é", "ABCDE"):
            with self.subTest(line=line), self.assertRaises(ValueError):
                transport.write_line(line)
        self.assertEqual(port.written, b"")

    def test_reports_read_write_and_zero_progress_failures(self):
        transport = SerialLineTransport(FakePort(OSError("read disconnected")))
        with self.assertRaisesRegex(SerialTransportError, "read disconnected"):
            transport.read_lines()
        transport = SerialLineTransport(FakePort(write_limit=OSError("write disconnected")))
        with self.assertRaisesRegex(SerialTransportError, "write disconnected"):
            transport.write_line("PING")
        transport = SerialLineTransport(FakePort(write_limit=0))
        with self.assertRaisesRegex(SerialTransportError, "progress"):
            transport.write_line("PING")

    def test_missing_discovery_directory_has_no_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(list_stable_serial_paths(Path(directory) / "missing"), [])


if __name__ == "__main__":
    unittest.main()
