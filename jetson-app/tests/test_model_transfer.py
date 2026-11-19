"""Loopback-only contract tests: no model runtime, camera or serial access."""
import hashlib
import http.client
import json
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

try:
    import model_transfer as transfer
except ImportError:
    transfer = None


class ModelTransferTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(transfer, 'LAN model receiver is not implemented')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.events = []
        self.received = []
        self.arrived = threading.Event()
        self.now = [100.0]

        def received(path, mode, job_id):
            self.received.append((path, mode, job_id, threading.current_thread().ident))
            self.arrived.set()

        self.receiver = transfer.ModelReceiver(Path(self.temp.name), received,
                                               self.events.append, host='127.0.0.1',
                                               port=0, discovery_port=0,
                                               clock=lambda: self.now[0])
        self.port = self.receiver.start()
        self.addCleanup(self.receiver.close)

    def request(self, method, path, data=None, token=None, headers=None):
        supplied = dict(headers or {})
        if token is not None:
            supplied['Authorization'] = 'Bearer ' + token
        conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=3)
        try:
            conn.request(method, path, body=data, headers=supplied)
            response = conn.getresponse()
            return response.status, json.loads(response.read().decode('utf-8'))
        finally:
            conn.close()

    def discovery(self):
        return transfer.discover_receivers(timeout=.15, address='127.0.0.1',
                                            discovery_port=self.receiver.discovery_port)

    def allow(self):
        status = self.receiver.allow(seconds=300)
        self.assertNotIn('token', status)
        found = self.discovery()
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]['port'], self.port)
        return found[0]

    def upload(self, endpoint, data=b'ONNX fixture bytes', mode='legacy_features', **overrides):
        headers = {'X-Filename': 'new model.onnx', 'X-Model-Mode': mode,
                   'X-SHA256': hashlib.sha256(data).hexdigest(),
                   'Content-Length': str(len(data))}
        headers.update(overrides)
        return self.request('POST', '/v1/upload', data, endpoint['token'], headers)

    def test_closed_admission_has_no_discovery_or_upload(self):
        self.assertEqual(self.discovery(), [])
        status, body = self.request('POST', '/v1/upload', b'no', 'untrusted')
        self.assertEqual(status, 403)
        self.assertEqual(list(Path(self.temp.name).iterdir()), [])
        self.assertNotIn('untrusted', json.dumps(body))

    def test_success_is_single_upload_with_generated_path_and_job_poll(self):
        endpoint = self.allow()
        data = b'ONNX fixture bytes'
        status, job = self.upload(endpoint, data)
        self.assertEqual(status, 202)
        self.assertTrue(self.arrived.wait(1))
        path, mode, job_id, thread_id = self.received[0]
        self.assertIsInstance(path, Path)
        self.assertEqual(path.parent, Path(self.temp.name).resolve())
        self.assertNotEqual(path.name, 'new model.onnx')
        self.assertEqual(path.suffix, '.onnx')
        self.assertEqual(path.read_bytes(), data)
        self.assertEqual(mode, 'legacy_features')
        self.assertEqual(job_id, job['job_id'])
        self.assertNotEqual(thread_id, threading.current_thread().ident)
        self.assertEqual(self.discovery(), [])
        self.assertEqual(self.upload(endpoint)[0], 403)
        self.receiver.update_job(job_id, 'ready', '转换完成', {'classes': 8})
        status, result = self.request('GET', '/v1/jobs/' + job_id, token=endpoint['token'])
        self.assertEqual(status, 200)
        self.assertEqual(result['state'], 'ready')
        self.assertEqual(result['details'], {'classes': 8})
        self.assertNotIn(endpoint['token'], json.dumps(self.events))
        self.assertNotIn(endpoint['token'], json.dumps(self.receiver.status()))

    def test_expired_allowance_and_wrong_token_cannot_upload(self):
        endpoint = self.allow()
        self.assertEqual(self.request('POST', '/v1/upload', b'x', 'wrong')[0], 403)
        self.now[0] += 301
        self.assertEqual(self.discovery(), [])
        self.assertEqual(self.upload(endpoint)[0], 403)
        self.assertEqual(self.received, [])

    def test_sha_mismatch_leaves_no_partial_file_and_admission_can_retry(self):
        endpoint = self.allow()
        status, body = self.upload(endpoint, **{'X-SHA256': '0' * 64})
        self.assertEqual(status, 422)
        self.assertEqual(list(Path(self.temp.name).iterdir()), [])
        self.assertEqual(self.received, [])
        self.assertEqual(self.upload(endpoint, mode='raw_rgb')[0], 202)

    def test_rejects_traversal_non_onnx_mode_and_invalid_digest(self):
        endpoint = self.allow()
        for header, value in (('X-Filename', '../bad.onnx'), ('X-Filename', 'C:\\bad.onnx'),
                              ('X-Filename', 'bad.exe'), ('X-Model-Mode', 'activate'),
                              ('X-SHA256', 'not-a-sha')):
            with self.subTest(header=header, value=value):
                self.assertEqual(self.upload(endpoint, **{header: value})[0], 400)
        self.assertEqual(list(Path(self.temp.name).iterdir()), [])

    def test_oversized_and_invalid_lengths_are_rejected_before_body(self):
        endpoint = self.allow()
        for value, expected in ((str(512 * 1024 * 1024 + 1), 413), ('0', 400), ('-1', 400)):
            with self.subTest(length=value):
                self.assertEqual(self.upload(endpoint, **{'Content-Length': value})[0], expected)
        self.assertEqual(list(Path(self.temp.name).iterdir()), [])

    def test_poll_is_bound_to_job_token_and_unknown_endpoints_do_nothing(self):
        endpoint = self.allow()
        _, job = self.upload(endpoint)
        self.assertEqual(self.request('GET', '/v1/jobs/' + job['job_id'], token='wrong')[0], 403)
        self.assertEqual(self.request('GET', '/v1/jobs/' + '0' * 32, token=endpoint['token'])[0], 404)
        self.assertEqual(self.request('POST', '/activate', b'', endpoint['token'])[0], 404)
        self.assertEqual(len(self.received), 1)

    def test_discovery_requires_protocol_and_correlated_nonce(self):
        self.receiver.allow()
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(.15)
            sock.sendto(b'{"magic":"wrong","nonce":"abc"}', ('127.0.0.1', self.receiver.discovery_port))
            with self.assertRaises(socket.timeout):
                sock.recvfrom(4096)

    def test_sender_streams_file_and_polls_using_discovered_token(self):
        endpoint = self.allow()
        source = Path(self.temp.name).parent / ('sender-' + str(self.port) + '.onnx')
        source.write_bytes(b'not parsed by the transport layer')
        self.addCleanup(lambda: source.unlink() if source.exists() else None)
        progress = []
        job = transfer.send_model(endpoint, source, 'raw_rgb',
                                  on_progress=lambda sent, total: progress.append((sent, total)))
        self.assertTrue(self.arrived.wait(1))
        self.receiver.update_job(job['job_id'], 'ready', 'ok')
        result = transfer.get_job(endpoint, job['job_id'])
        self.assertEqual(result['state'], 'ready')
        self.assertEqual(progress[-1], (source.stat().st_size, source.stat().st_size))
        self.assertEqual(self.received[0][1], 'raw_rgb')

    def test_sender_rejects_non_lan_ip_and_non_onnx_before_network(self):
        with self.assertRaises(ValueError):
            transfer.discover_receivers(timeout=.01, address='8.8.8.8')
        endpoint = self.allow()
        source = Path(self.temp.name) / 'not-a-model.txt'
        source.write_bytes(b'x')
        with self.assertRaises(ValueError):
            transfer.send_model(endpoint, source, 'raw_rgb')
        self.assertEqual(self.received, [])

    def test_second_upload_cannot_race_an_inflight_first_upload(self):
        endpoint = self.allow()
        conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=3)
        self.addCleanup(conn.close)
        data = b'x' * 4096
        conn.putrequest('POST', '/v1/upload')
        for name, value in (('Authorization', 'Bearer ' + endpoint['token']),
                            ('X-Filename', 'one.onnx'), ('X-Model-Mode', 'raw_rgb'),
                            ('X-SHA256', hashlib.sha256(data).hexdigest()),
                            ('Content-Length', str(len(data)))):
            conn.putheader(name, value)
        conn.endheaders()
        until = time.monotonic() + 1
        while not self.receiver.status().get('uploading') and time.monotonic() < until:
            time.sleep(.005)
        self.assertTrue(self.receiver.status()['uploading'])
        self.assertEqual(self.upload(endpoint)[0], 409)
        conn.send(data)
        response = conn.getresponse()
        self.assertEqual(response.status, 202)
        response.read()
        self.assertTrue(self.arrived.wait(1))
        self.assertEqual(len(self.received), 1)

    def test_non_ascii_capability_is_rejected_without_crashing_handler(self):
        self.allow()
        status, body = self.request('POST', '/v1/upload', b'x', 'é' * 40)
        self.assertEqual(status, 403)
        self.assertEqual(body['error'], 'admission_closed_or_unauthorized')

    def test_expiry_during_upload_removes_partial_and_does_not_queue(self):
        endpoint = self.allow()
        conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=3)
        self.addCleanup(conn.close)
        data = b'x' * 1024
        conn.putrequest('POST', '/v1/upload')
        for name, value in (('Authorization', 'Bearer ' + endpoint['token']),
                            ('X-Filename', 'one.onnx'), ('X-Model-Mode', 'raw_rgb'),
                            ('X-SHA256', hashlib.sha256(data).hexdigest()),
                            ('Content-Length', str(len(data)))):
            conn.putheader(name, value)
        conn.endheaders()
        until = time.monotonic() + 1
        while not self.receiver.status()['uploading'] and time.monotonic() < until:
            time.sleep(.005)
        self.assertTrue(self.receiver.status()['uploading'])
        self.now[0] += 301
        conn.send(data)
        response = conn.getresponse()
        self.assertEqual(response.status, 403)
        response.read()
        self.assertEqual(list(Path(self.temp.name).iterdir()), [])
        self.assertEqual(self.received, [])

    def test_callback_failure_is_failed_job_without_exception_or_token_leak(self):
        endpoint = self.allow()

        def broken(*args):
            raise RuntimeError('private ' + endpoint['token'])

        self.receiver.on_received = broken
        _, job = self.upload(endpoint)
        status, body = self.request('GET', '/v1/jobs/' + job['job_id'], token=endpoint['token'])
        self.assertEqual(status, 200)
        self.assertEqual(body['state'], 'failed')
        self.assertNotIn('private', json.dumps(body))
        self.assertNotIn(endpoint['token'], json.dumps(self.events))

    def test_allow_waits_for_terminal_job_and_rotates_only_upload_token(self):
        first = self.allow()
        _, job = self.upload(first)
        with self.assertRaises(RuntimeError):
            self.receiver.allow()
        self.receiver.update_job(job['job_id'], 'ready', first['token'], {'echo': first['token']})
        body = transfer.get_job(dict(first, host='127.0.0.1'), job['job_id'])
        self.assertNotIn(first['token'], json.dumps(body))
        second = self.allow()
        self.assertNotEqual(first['token'], second['token'])
        self.assertEqual(self.upload(first)[0], 403)
        self.assertEqual(self.request('GET', '/v1/jobs/' + job['job_id'], token=first['token'])[0], 200)

    def test_sender_contract_validation_calls_inspector_before_network(self):
        try:
            import windows_model_sender as sender
        except ImportError:
            sender = None
        self.assertIsNotNone(sender, 'Windows sender GUI is not implemented')
        source = Path(self.temp.name) / 'model.onnx'
        source.write_bytes(b'graph placeholder')
        inspector = mock.Mock(return_value={'classes': 8, 'input_shape': [1, 3, 320, 320]})
        with mock.patch.dict('sys.modules', {'onnx_contract': mock.Mock(inspect_yolo_onnx=inspector)}):
            result = sender.validate_model(source)
        inspector.assert_called_once()
        self.assertEqual(result['classes'], 8)
        self.assertEqual(self.received, [])

    def test_sender_diagnose_and_self_test_are_explicit_modes(self):
        try:
            import windows_model_sender as sender
        except ImportError:
            sender = None
        self.assertIsNotNone(sender, 'Windows sender GUI is not implemented')
        self.assertTrue(callable(sender.run_self_test))
        self.assertTrue(callable(sender.diagnose))
        self.assertIn('--self-test', sender.argument_parser().format_help())
        self.assertIn('--diagnose', sender.argument_parser().format_help())

    def test_missing_duplicate_length_and_chunked_framing_are_rejected(self):
        endpoint = self.allow()
        for framing, expected in (([], 411), ([('Content-Length', '2'), ('Content-Length', '2')], 400),
                                   ([('Transfer-Encoding', 'chunked')], 400)):
            with self.subTest(framing=framing):
                conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=3)
                try:
                    conn.putrequest('POST', '/v1/upload')
                    conn.putheader('Authorization', 'Bearer ' + endpoint['token'])
                    for name, value in framing:
                        conn.putheader(name, value)
                    conn.endheaders()
                    reply = conn.getresponse()
                    self.assertEqual(reply.status, expected)
                    reply.read()
                finally:
                    conn.close()
        self.assertEqual(list(Path(self.temp.name).iterdir()), [])

    def test_truncated_body_is_cleaned_before_error_response(self):
        endpoint = self.allow()
        conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=3)
        try:
            conn.putrequest('POST', '/v1/upload')
            for name, value in (('Authorization', 'Bearer ' + endpoint['token']),
                                ('X-Filename', 'one.onnx'), ('X-Model-Mode', 'raw_rgb'),
                                ('X-SHA256', '0' * 64), ('Content-Length', '100')):
                conn.putheader(name, value)
            conn.endheaders()
            conn.send(b'short')
            conn.sock.shutdown(socket.SHUT_WR)
            reply = conn.getresponse()
            self.assertEqual(reply.status, 400)
            self.assertEqual(json.loads(reply.read())['error'], 'upload_interrupted')
        finally:
            conn.close()
        self.assertEqual(list(Path(self.temp.name).iterdir()), [])
        self.assertFalse(self.receiver.status()['uploading'])
        self.assertEqual(self.received, [])

    def test_extreme_length_header_is_rejected_without_integer_overflow(self):
        endpoint = self.allow()
        status, _ = self.upload(endpoint, **{'Content-Length': '9' * 5000})
        self.assertEqual(status, 400)

    @unittest.skipUnless(sys.platform == 'win32', 'Native Tk GUI is verified on the Windows sender host')
    def test_tk_scan_results_survive_the_busy_listbox_state(self):
        import tkinter as tk
        import windows_model_sender as sender
        root = tk.Tk()
        root.withdraw()
        app = sender.ModelSenderApp(root)
        try:
            app.set_busy(True)
            app.messages.put(('receivers', [{'host': '127.0.0.1', 'expires_in': 300}]))
            app.messages.put(('idle', None))
            app.drain_messages()
            self.assertEqual(app.listbox.size(), 1)
            self.assertEqual(app.listbox.curselection(), (0,))
        finally:
            app.close()


if __name__ == '__main__':
    unittest.main()
