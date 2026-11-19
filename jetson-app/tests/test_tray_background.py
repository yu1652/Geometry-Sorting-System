"""An opt-in tray reference remains bound to its camera and pixel crop."""
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np

try:
    from tray_background import TrayBackgroundReference
except ImportError:
    TrayBackgroundReference = None


class TrayBackgroundReferenceTests(unittest.TestCase):
    source = '/dev/v4l/by-path/test-top-camera'
    roi = (16, 0, 519, 479)

    def setUp(self):
        self.assertIsNotNone(TrayBackgroundReference, 'Tray reference storage is missing')
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'tray_background.npz'
        self.frame = np.zeros((480, 640, 3), np.uint8)
        self.frame[10, 20] = [12, 34, 56]

    def reference(self):
        return TrayBackgroundReference(self.frame, self.source, self.roi)

    def archive(self, **updates):
        values = dict(version=np.array(1, np.int64), frame=self.frame,
                      source_top=np.array(self.source), roi=np.array(self.roi, np.int64))
        values.update(updates)
        np.savez_compressed(str(self.path), **values)

    def test_constructor_and_frame_property_never_share_mutable_image_storage(self):
        reference = self.reference()
        self.frame[10, 20] = 255
        np.testing.assert_array_equal(reference.frame[10, 20], [12, 34, 56])
        returned = reference.frame
        returned[10, 20] = 0
        np.testing.assert_array_equal(reference.frame[10, 20], [12, 34, 56])
        self.assertEqual(reference.source_top, self.source)
        self.assertEqual(reference.roi, self.roi)

    def test_roundtrip_preserves_full_frame_and_exact_camera_identity(self):
        self.reference().save(self.path)
        restored = TrayBackgroundReference.load(
            self.path, expected_source_top=self.source, expected_roi=self.roi)
        np.testing.assert_array_equal(restored.frame, self.frame)
        self.assertEqual(restored.frame.dtype, np.uint8)
        self.assertEqual(restored.source_top, self.source)
        self.assertEqual(restored.roi, self.roi)
        # The storage artifact contains only non-pickle NumPy members.
        with np.load(str(self.path), allow_pickle=False) as archive:
            self.assertEqual(set(archive.files), {'version', 'frame', 'source_top', 'roi'})
            self.assertTrue(all(not archive[key].dtype.hasobject for key in archive.files))

    def test_camera_and_roi_mismatches_never_reuse_reference(self):
        self.reference().save(self.path)
        for settings in ({'expected_source_top': '/different-camera'},
                         {'expected_roi': (17, 0, 518, 479)},
                         {'expected_roi': (16., 0, 519, 479)}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                TrayBackgroundReference.load(self.path, **settings)

    def test_frame_must_be_full_raw_uint8_bgr(self):
        for frame in (None, [[0]], np.zeros((480, 640), np.uint8),
                      np.zeros((479, 640, 3), np.uint8),
                      np.zeros((480, 640, 3), np.uint16),
                      np.full((480, 640, 3), np.nan, np.float32)):
            with self.subTest(shape=getattr(frame, 'shape', None)), self.assertRaises(ValueError):
                TrayBackgroundReference(frame, self.source, self.roi)

    def test_source_and_roi_metadata_reject_invalid_identities(self):
        for source in (None, '', '  ', 3, 'camera\x00other', 'x' * 4097):
            with self.subTest(source_type=type(source).__name__), self.assertRaises(ValueError):
                TrayBackgroundReference(self.frame, source, self.roi)
        for roi in (None, (), (16, 0, 519), (16, 0, 519, 479, 1),
                    (16., 0, 519, 479), (True, 0, 519, 479), (-1, 0, 10, 10),
                    (0, 0, 0, 10), (0, 0, 641, 480), (0, 1, 640, 480),
                    (0, np.nan, 640, 480)):
            with self.subTest(roi=roi), self.assertRaises(ValueError):
                TrayBackgroundReference(self.frame, self.source, roi)

    def test_load_rejects_wrong_version_dtype_shape_and_nonfinite_metadata(self):
        for updates in ({'version': np.array(2, np.int64)},
                        {'version': np.array([1], np.int64)},
                        {'version': np.array(np.nan)},
                        {'frame': self.frame.astype(np.float32)},
                        {'frame': np.zeros((480, 640, 4), np.uint8)},
                        {'roi': np.array([16., 0., 519., 479.])},
                        {'roi': np.array([16., np.inf, 519., 479.])},
                        {'source_top': np.array([self.source])}):
            with self.subTest(keys=list(updates)):
                self.archive(**updates)
                with self.assertRaises(ValueError):
                    TrayBackgroundReference.load(self.path)

    def test_pickle_members_and_unexpected_members_are_rejected(self):
        for updates in ({'source_top': np.array({'source': self.source}, dtype=object)},
                        {'unrelated': np.array([1])}):
            self.archive(**updates)
            with self.assertRaises(ValueError):
                TrayBackgroundReference.load(self.path)

    def test_duplicate_members_are_rejected(self):
        self.archive()
        with ZipFile(str(self.path), 'a') as archive:
            with self.assertWarns(UserWarning):
                archive.writestr('roi.npy', b'not a second valid member')
        with self.assertRaises(ValueError):
            TrayBackgroundReference.load(self.path)

    def test_compressed_and_uncompressed_archive_sizes_are_bounded(self):
        self.path.write_bytes(b'x' * 2_000_001)
        with self.assertRaises(ValueError):
            TrayBackgroundReference.load(self.path)
        with ZipFile(str(self.path), 'w', ZIP_DEFLATED) as archive:
            archive.writestr('frame.npy', b'0' * 2_000_001)
        self.assertLess(self.path.stat().st_size, 2_000_000)
        with self.assertRaises(ValueError):
            TrayBackgroundReference.load(self.path)

    def test_forged_huge_array_header_is_rejected_before_allocation(self):
        self.archive()
        header = BytesIO()
        np.lib.format.write_array_header_1_0(
            header, {'descr': '|u1', 'fortran_order': False, 'shape': (10**12, 640, 3)})
        with ZipFile(str(self.path)) as archive:
            members = {name: archive.read(name) for name in archive.namelist()}
        members['frame.npy'] = header.getvalue()
        with ZipFile(str(self.path), 'w', ZIP_DEFLATED) as archive:
            for name, raw in members.items():
                archive.writestr(name, raw)
        with self.assertRaises(ValueError):
            TrayBackgroundReference.load(self.path)

    def test_corrupt_archives_raise_value_error(self):
        self.path.write_bytes(b'not a NumPy archive')
        with self.assertRaises(ValueError):
            TrayBackgroundReference.load(self.path)

    def test_replace_failure_keeps_previous_file_and_cleans_temporary_file(self):
        self.reference().save(self.path)
        before = self.path.read_bytes()
        with patch('tray_background.os.replace', side_effect=OSError('replace failed')):
            with self.assertRaises(OSError):
                self.reference().save(self.path)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_encoding_failure_keeps_previous_file_and_cleans_temporary_file(self):
        self.reference().save(self.path)
        before = self.path.read_bytes()

        def failed_encode(stream, **arrays):
            stream.write(b'partial')
            raise OSError('write failed')

        with patch('tray_background.np.savez_compressed', side_effect=failed_encode):
            with self.assertRaises(OSError):
                self.reference().save(self.path)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])


if __name__ == '__main__':
    unittest.main()
