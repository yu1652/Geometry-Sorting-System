"""Conversion reuse is content/environment bound and cannot publish failed work."""
import hashlib
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from side_model_store import SideCandidate, SideModelStore, atomic_json


NAMES = ['si_leng_zhu', 'liu_leng_zhu']
ENV = {'schema': 1, 'tensorrt': '8.2.1', 'cuda_runtime': 10020,
       'hardware': 'Xavier NX', 'flags': ['--fp16', '--workspace=1024', '--buildOnly']}


def candidate_at(stage):
    path = stage / 'manifest.json'
    return SideCandidate(path, json.loads(path.read_text(encoding='utf8')))


class Backend:
    def __init__(self, bad=False):
        self.bad = bad
        self.closed = False

    def infer(self, value):
        output = np.zeros((1, 4 + len(NAMES), 2100), np.float32)
        if self.bad:
            output[0, 0, 0] = np.nan
        return output

    def close(self):
        self.closed = True


class ConversionCacheTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.store = SideModelStore(self.root / 'app')
        self.source = self.root / 'new.onnx'
        self.source.write_bytes(b'validated-onnx')
        self.contract = {'names': NAMES, 'input_shape': [1, 3, 320, 320],
                         'output_shape': [1, 6, 2100], 'sha256': hashlib.sha256(self.source.read_bytes()).hexdigest()}

    def candidate(self, mode='raw_rgb', fingerprint=ENV, onnx_sha=None, engine=b'cached-engine'):
        stage = self.store._new_version()
        (stage / 'shape_fp16.engine').write_bytes(engine)
        record = {'schema': 1, 'class_names': NAMES, 'input_shape': [1, 3, 320, 320],
                  'output_shape': [1, 6, 2100], 'preprocess': mode, 'engine': 'shape_fp16.engine',
                  'engine_sha256': hashlib.sha256(engine).hexdigest(),
                  'source': {'onnx_sha256': onnx_sha or self.contract['sha256']}}
        if fingerprint is not None:
            record['conversion_fingerprint'] = fingerprint
        atomic_json(stage / 'manifest.json', record)
        return stage

    def patches(self):
        return patch.multiple(self.store, inspect_onnx=lambda source: dict(self.contract),
                              conversion_fingerprint=lambda: dict(ENV), _running_trtexec=lambda: [])

    def build(self, argv, **kwargs):
        target = next(value.split('=', 1)[1] for value in argv if value.startswith('--saveEngine='))
        Path(target).write_bytes(b'new-engine')
        return SimpleNamespace(returncode=0)

    def test_matching_cache_native_smoke_reuses_without_trtexec_or_active_write(self):
        stage = self.candidate()
        backend = Backend()
        with self.patches(), patch('side_model_store.create_backend', return_value=backend), \
             patch('side_model_store.subprocess.run') as build:
            result = self.store.prepare_onnx(self.source, 'raw_rgb')
        self.assertEqual(result.manifest_path.parent, stage)
        build.assert_not_called()
        self.assertTrue(backend.closed)
        self.assertFalse(self.store.active_path.exists())

    def test_legacy_manual_candidate_without_fingerprint_can_be_adopted(self):
        stage = self.candidate(fingerprint=None)
        before = (stage / 'manifest.json').read_bytes()
        with self.patches(), patch('side_model_store.create_backend', return_value=Backend()), \
             patch('side_model_store.subprocess.run') as build:
            result = self.store.prepare_onnx(self.source, 'raw_rgb')
        self.assertEqual(result.manifest_path.parent, stage)
        self.assertEqual((stage / 'manifest.json').read_bytes(), before)
        build.assert_not_called()

    def test_fingerprint_mismatch_forces_new_stage_even_if_old_engine_looks_valid(self):
        for field, value in (('tensorrt', 'different'), ('cuda_runtime', 12000),
                             ('hardware', 'other GPU'), ('flags', ['--fp16', '--workspace=512'])):
            with self.subTest(field=field):
                self.store = SideModelStore(self.root / field)
                old = self.candidate(fingerprint=dict(ENV, **{field: value}))
                with self.patches(), patch('side_model_store.create_backend', return_value=Backend()), \
                     patch('side_model_store.subprocess.run', side_effect=self.build) as build:
                    result = self.store.prepare_onnx(self.source, 'raw_rgb')
                self.assertNotEqual(result.manifest_path.parent, old)
                self.assertEqual(result.manifest['conversion_fingerprint'], ENV)
                self.assertEqual(build.call_count, 1)

    def test_preprocess_or_content_change_cannot_reuse(self):
        self.candidate(mode='legacy_features')
        self.candidate(onnx_sha='0' * 64)
        with self.patches(), patch('side_model_store.create_backend', return_value=Backend()), \
             patch('side_model_store.subprocess.run', side_effect=self.build) as build:
            self.store.prepare_onnx(self.source, 'raw_rgb')
        self.assertEqual(build.call_count, 1)

    def test_hash_damaged_engine_is_skipped(self):
        old = self.candidate()
        (old / 'shape_fp16.engine').write_bytes(b'tampered')
        with self.patches(), patch('side_model_store.create_backend', return_value=Backend()), \
             patch('side_model_store.subprocess.run', side_effect=self.build) as build:
            result = self.store.prepare_onnx(self.source, 'raw_rgb')
        self.assertNotEqual(result.manifest_path.parent, old)
        self.assertEqual(build.call_count, 1)

    def test_native_smoke_failure_skips_cache_and_closes_backend(self):
        old = self.candidate()
        broken = Backend(bad=True)
        with self.patches(), patch('side_model_store.create_backend', side_effect=[broken, Backend()]), \
             patch('side_model_store.subprocess.run', side_effect=self.build) as build:
            result = self.store.prepare_onnx(self.source, 'raw_rgb')
        self.assertNotEqual(result.manifest_path.parent, old)
        self.assertTrue(broken.closed)
        self.assertEqual(build.call_count, 1)

    def test_build_failure_preserves_active_and_existing_candidates(self):
        old = self.candidate(mode='legacy_features')
        self.store.active_path.parent.mkdir(parents=True)
        self.store.active_path.write_bytes(b'{"existing":"unchanged"}')
        before = (old / 'manifest.json').read_bytes()
        with self.patches(), patch('side_model_store.subprocess.run', return_value=SimpleNamespace(returncode=1)):
            with self.assertRaises(ValueError):
                self.store.prepare_onnx(self.source, 'raw_rgb')
        self.assertEqual(self.store.active_path.read_bytes(), b'{"existing":"unchanged"}')
        self.assertEqual((old / 'manifest.json').read_bytes(), before)

    def test_existing_manual_trtexec_is_not_killed_or_joined(self):
        with self.patches(), patch.object(self.store, '_running_trtexec', return_value=[999]), \
             patch('side_model_store.subprocess.run') as build:
            with self.assertRaisesRegex(ValueError, 'trtexec'):
                self.store.prepare_onnx(self.source, 'raw_rgb')
        build.assert_not_called()
        self.assertFalse(self.store.active_path.exists())

    def test_second_store_cannot_build_while_os_file_lock_is_held(self):
        other = SideModelStore(self.store.root)
        with self.store._conversion_lock(), self.patches(), \
             patch.object(other, 'inspect_onnx', return_value=self.contract), \
             patch.object(other, 'conversion_fingerprint', return_value=ENV), \
             patch('side_model_store.subprocess.run') as build:
            with self.assertRaisesRegex(ValueError, '转换'):
                other.prepare_onnx(self.source, 'raw_rgb')
        build.assert_not_called()

    def test_proc_scan_only_reports_real_trtexec_executable_for_this_user(self):
        proc = self.root / 'proc'
        process = proc / '999999'
        process.mkdir(parents=True)
        (process / 'cmdline').write_bytes(b'/usr/src/tensorrt/bin/trtexec\0--onnx=test.onnx\0')
        other = proc / '999998'
        other.mkdir()
        (other / 'cmdline').write_bytes(b'/usr/bin/python3\0trtexec\0')
        self.assertEqual(self.store._running_trtexec(proc, process.stat().st_uid), [999999])
        self.assertEqual(self.store._running_trtexec(proc, process.stat().st_uid + 1), [])

    def test_native_fingerprint_includes_versions_device_builder_and_flags(self):
        class Function:
            def __init__(self, values):
                self.values = values
            def __call__(self, output, *args):
                output._obj.value = self.values[args[0]] if args else self.values
                return 0
        native = SimpleNamespace(cudaRuntimeGetVersion=Function(10020), cudaDriverGetVersion=Function(11040),
                                 cudaDeviceGetAttribute=Function({75: 7, 76: 2}))
        exe = self.root / 'trtexec'
        exe.write_bytes(b'controlled-builder-version')
        import side_model_store
        original_import = side_model_store.importlib.import_module
        def import_native(name, *args, **kwargs):
            return SimpleNamespace(__version__='8.2.1') if name == 'tensorrt' else original_import(name, *args, **kwargs)
        with patch('side_model_store.ctypes.CDLL', return_value=native), \
             patch('side_model_store.importlib.import_module', side_effect=import_native), \
             patch('side_model_store.shutil.which', return_value=str(exe)):
            value = self.store.conversion_fingerprint()
        self.assertEqual(value['tensorrt'], '8.2.1')
        self.assertEqual(value['cuda_runtime'], 10020)
        self.assertEqual(value['cuda_driver'], 11040)
        self.assertEqual(value['compute_capability'], [7, 2])
        self.assertEqual(value['trtexec_sha256'], hashlib.sha256(exe.read_bytes()).hexdigest())
        self.assertEqual(value['flags'], ENV['flags'])

    def test_repeat_after_build_is_a_cache_hit(self):
        with self.patches(), patch('side_model_store.create_backend', side_effect=lambda *a, **k: Backend()), \
             patch('side_model_store.subprocess.run', side_effect=self.build) as build:
            first = self.store.prepare_onnx(self.source, 'raw_rgb')
            second = self.store.prepare_onnx(self.source, 'raw_rgb')
        self.assertEqual(first.manifest_path, second.manifest_path)
        self.assertEqual(build.call_count, 1)

    def test_changed_source_after_inspection_is_rejected_before_build(self):
        self.source.write_bytes(b'different-content')
        with self.patches(), patch('side_model_store.subprocess.run') as build:
            with self.assertRaisesRegex(ValueError, 'ONNX'):
                self.store.prepare_onnx(self.source, 'raw_rgb')
        build.assert_not_called()

    def test_rollback_validates_previous_and_only_replaces_active_pointer(self):
        old = self.candidate(engine=b'previous')
        current = self.candidate(engine=b'current')
        from side_model_store import SideCandidate
        self.store.activate(candidate_at(old))
        self.store.activate(candidate_at(current))
        calibration = self.store.root / 'data/live_calibration.json'
        calibration.write_bytes(b'unchanged-nine-points')
        with patch('side_model_store.create_backend', return_value=Backend()):
            result = self.store.rollback_previous()
        self.assertEqual(result.manifest_path.parent, old)
        self.assertEqual(self.store.read_active()['version'], old.name)
        self.assertEqual(calibration.read_bytes(), b'unchanged-nine-points')

    def test_damaged_previous_or_missing_previous_does_not_change_pointer(self):
        old = self.candidate(engine=b'previous')
        current = self.candidate(engine=b'current')
        from side_model_store import SideCandidate
        self.store.activate(candidate_at(old))
        with self.assertRaises(ValueError):
            self.store.rollback_previous()
        self.store.activate(candidate_at(current))
        before = self.store.active_path.read_bytes()
        (old / 'shape_fp16.engine').write_bytes(b'damaged')
        with self.assertRaises(ValueError):
            self.store.rollback_previous()
        self.assertEqual(self.store.active_path.read_bytes(), before)

    def test_rollback_does_not_clobber_a_pointer_changed_during_native_trial(self):
        old = self.candidate(engine=b'previous')
        current = self.candidate(engine=b'current')
        from side_model_store import SideCandidate
        self.store.activate(candidate_at(old))
        self.store.activate(candidate_at(current))
        replacement = b'{"changed":"by another manager"}'
        backend = Backend()
        original = backend.infer
        def trial(value):
            self.store.active_path.write_bytes(replacement)
            return original(value)
        backend.infer = trial
        with patch('side_model_store.create_backend', return_value=backend):
            with self.assertRaisesRegex(ValueError, '变化'):
                self.store.rollback_previous()
        self.assertEqual(self.store.active_path.read_bytes(), replacement)

    def test_finish_does_not_publish_engine_changed_during_smoke(self):
        stage = self.store._new_version()
        engine = stage / 'shape_fp16.engine'
        engine.write_bytes(b'the-smoked-engine')
        backend = Backend()
        original = backend.infer
        def trial(value):
            engine.write_bytes(b'not-the-smoked-engine')
            return original(value)
        backend.infer = trial
        with patch('side_model_store.create_backend', return_value=backend):
            with self.assertRaisesRegex(ValueError, '变化'):
                self.store._finish(stage, NAMES, 'raw_rgb', {}, lambda message: None)
        self.assertFalse((stage / 'manifest.json').exists())
        self.assertFalse(self.store.active_path.exists())

    def test_activation_cannot_accept_manifest_replaced_after_candidate_trial(self):
        stage = self.candidate()
        from side_model_store import SideCandidate
        path = stage / 'manifest.json'
        candidate = SideCandidate(path, json.loads(path.read_text(encoding='utf8')))
        changed = dict(candidate.manifest, preprocess='legacy_features')
        atomic_json(path, changed)
        with self.assertRaisesRegex(ValueError, '变化'):
            self.store.activate(candidate)
        self.assertFalse(self.store.active_path.exists())

    def test_reactivating_cached_current_version_keeps_true_previous_without_writing(self):
        previous = self.candidate(engine=b'previous-A')
        current = self.candidate(engine=b'current-B')
        self.store.activate(candidate_at(previous))
        self.store.activate(candidate_at(current))
        before = self.store.active_path.read_bytes()
        with patch('side_model_store.atomic_json') as publish, \
             patch.object(self.store, 'read_active', wraps=self.store.read_active) as validate:
            self.store.activate(candidate_at(current))
        publish.assert_not_called()
        validate.assert_called_once_with()
        self.assertEqual(self.store.active_path.read_bytes(), before)
        pointer = json.loads(before.decode('utf8'))
        self.assertEqual(pointer['version'], current.name)
        self.assertEqual(pointer['previous']['version'], previous.name)
        with patch('side_model_store.create_backend', return_value=Backend()):
            self.store.rollback_previous()
        self.assertEqual(self.store.read_active()['version'], previous.name)


if __name__ == '__main__':
    unittest.main()
