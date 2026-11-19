import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from install_model_tools import FILES, install


class InstallTests(unittest.TestCase):
    def fixture(self, directory):
        home = Path(directory)
        payload = home / 'payload'
        target = home / 'board' / 'jetson-app'
        payload.mkdir()
        (target / 'data').mkdir(parents=True)
        (target / 'run_jp4.sh').write_text('existing launcher')
        (target / 'side_model_store.py').write_text('old model store')
        (target / 'data' / 'live_side_model.json').write_text('active model untouched')
        (target / 'data' / 'live_calibration.json').write_text('calibration untouched')
        hashes = {}
        for name in FILES:
            content = ('new payload ' + name).encode('utf8')
            (payload / name).write_bytes(content)
            hashes[name] = hashlib.sha256(content).hexdigest()
        (payload / 'payload-manifest.json').write_text(json.dumps({'files': hashes}))
        return home, payload, target

    def test_install_backs_up_code_and_preserves_model_and_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            home, payload, target = self.fixture(directory)
            result = install(payload, target, home=home)
            self.assertEqual((target / 'data' / 'live_side_model.json').read_text(), 'active model untouched')
            self.assertEqual((target / 'data' / 'live_calibration.json').read_text(), 'calibration untouched')
            backup = Path(result['backup'])
            self.assertEqual((backup / 'side_model_store.py').read_text(), 'old model store')
            self.assertTrue(Path(result['shortcut']).is_file())
            self.assertIn(str(target), Path(result['shortcut']).read_text(encoding='utf8'))

    def test_bad_payload_is_rejected_before_any_code_is_changed(self):
        with tempfile.TemporaryDirectory() as directory:
            home, payload, target = self.fixture(directory)
            (payload / 'side_model_manager.py').write_text('tampered')
            with self.assertRaises(ValueError):
                install(payload, target, home=home)
            self.assertEqual((target / 'side_model_store.py').read_text(), 'old model store')

    def test_install_target_must_be_inside_user_home(self):
        with tempfile.TemporaryDirectory() as directory:
            home, payload, target = self.fixture(directory)
            with self.assertRaises(ValueError):
                install(payload, target, home=home / 'some-other-home')

    def test_desktop_write_failure_restores_all_replaced_code(self):
        import install_model_tools
        original_write = install_model_tools._write
        with tempfile.TemporaryDirectory() as directory:
            home, payload, target = self.fixture(directory)
            def fail_desktop(path, content, executable=False):
                if path.name == 'ONNX模型转换器.desktop':
                    raise OSError('desktop not writable')
                return original_write(path, content, executable)
            with patch('install_model_tools._write', side_effect=fail_desktop):
                with self.assertRaises(OSError):
                    install(payload, target, home=home)
            self.assertEqual((target / 'side_model_store.py').read_text(), 'old model store')
            self.assertFalse((target / 'onnx_contract.py').exists())
