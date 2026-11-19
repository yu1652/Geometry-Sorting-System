"""One-shot LAN ONNX inbox and streaming client; Python 3.6 standard library.

The receiver never imports models, activates a model, or executes a command.
Callbacks run on background threads: Qt callers must emit a queued signal.
Admission tokens are short-lived capabilities, not account passwords.
"""
import hashlib
import hmac
import http.client
from http.server import BaseHTTPRequestHandler, HTTPServer
import ipaddress
import json
import math
from pathlib import Path
import re
import secrets
import socket
from socketserver import ThreadingMixIn
import threading
import time
import uuid


MAGIC = 'YLERS_MODEL_TRANSFER_V1'
DISCOVERY_PORT = 37888
HTTP_PORT = 8788
MAX_MODEL_BYTES = 512 * 1024 * 1024
MODES = ('legacy_features', 'raw_rgb')
_DIGEST = re.compile(r'^[0-9a-fA-F]{64}$')
_JOB_ID = re.compile(r'^[0-9a-f]{32}$')
_TERMINAL = ('ready', 'failed', 'cancelled')
_JOB_STATES = ('queued', 'building') + _TERMINAL
_PRIVATE_NETWORKS = tuple(ipaddress.ip_network(value) for value in
                          ('10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16',
                           '127.0.0.0/8', '169.254.0.0/16'))


class _UploadRejected(Exception):
    def __init__(self, code, error):
        self.code, self.error = code, error


def _lan_ip(value):
    try:
        address = ipaddress.IPv4Address(str(value))
    except ipaddress.AddressValueError:
        raise ValueError('请输入板子当前的局域网 IPv4 地址')
    if not any(address in network for network in _PRIVATE_NETWORKS):
        raise ValueError('仅支持同一局域网内的 IPv4 地址')
    return str(address)


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8')


class _ThreadedServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def handle_error(self, request, client_address):
        # Do not log request headers, capabilities or callback exceptions.
        pass


class ModelReceiver:
    def __init__(self, inbox, on_received, on_event=None, host='0.0.0.0',
                 port=HTTP_PORT, discovery_port=DISCOVERY_PORT, clock=None):
        if not callable(on_received):
            raise ValueError('on_received callback is required')
        self.inbox = Path(inbox).resolve()
        self.on_received = on_received
        self.on_event = on_event
        self.host = host
        self.port = int(port)
        self.discovery_port = int(discovery_port)
        self.receiver_id = uuid.uuid4().hex
        self._clock = clock or time.monotonic
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._http = None
        self._udp = None
        self._threads = []
        self._connections = set()
        self._token = None
        self._expires = 0.0
        self._uploading = False
        self._jobs = {}

    def start(self):
        with self._lock:
            if self._http is not None:
                return self.port
            if self._stop.is_set():
                raise RuntimeError('Receiver has closed')
            self.inbox.mkdir(parents=True, exist_ok=True)
            receiver = self

            class Handler(BaseHTTPRequestHandler):
                protocol_version = 'HTTP/1.0'

                def setup(self):
                    self.request.settimeout(30)
                    BaseHTTPRequestHandler.setup(self)
                    with receiver._lock:
                        receiver._connections.add(self.connection)

                def finish(self):
                    try:
                        BaseHTTPRequestHandler.finish(self)
                    finally:
                        with receiver._lock:
                            receiver._connections.discard(self.connection)

                def log_message(self, *args):
                    pass

                def reply(self, status, value):
                    # A tiny rejected POST may already be arriving while its
                    # headers are checked. Consume only bounded small bodies so
                    # Windows does not reset the socket before the JSON error
                    # reaches the client. Large/unknown bodies are never read.
                    lengths = self.headers.get_all('Content-Length') or []
                    if status >= 400 and self.command == 'POST' and len(lengths) == 1:
                        length_text = lengths[0]
                        if len(length_text) <= 6 and re.match(r'^[0-9]+$', length_text):
                            remaining = int(length_text) - getattr(self, '_body_read', 0)
                            if 0 < remaining <= 65536:
                                try:
                                    self.connection.settimeout(.2)
                                    self.rfile.read(remaining)
                                except OSError:
                                    pass
                                finally:
                                    self.connection.settimeout(30)
                    payload = _json_bytes(value)
                    self.send_response(status)
                    self.send_header('Content-Type', 'application/json; charset=utf-8')
                    self.send_header('Content-Length', str(len(payload)))
                    self.send_header('Cache-Control', 'no-store')
                    self.send_header('Connection', 'close')
                    self.end_headers()
                    self.wfile.write(payload)

                def capability(self):
                    values = self.headers.get_all('Authorization') or []
                    if len(values) != 1 or not values[0].startswith('Bearer '):
                        return ''
                    return values[0][7:]

                def do_GET(self):
                    if not self.path.startswith('/v1/jobs/'):
                        self.reply(404, {'error': 'unknown_endpoint'})
                        return
                    job_id = self.path[len('/v1/jobs/'):]
                    if not _JOB_ID.match(job_id):
                        self.reply(404, {'error': 'unknown_job'})
                        return
                    code, result = receiver._get_job(job_id, self.capability())
                    self.reply(code, result)

                def do_POST(self):
                    self._body_read = 0
                    if self.path != '/v1/upload':
                        self.reply(404, {'error': 'unknown_endpoint'})
                        return
                    token = self.capability()
                    with receiver._lock:
                        if not receiver._allowed(token):
                            self.reply(403, {'error': 'admission_closed_or_unauthorized'})
                            return
                    lengths = self.headers.get_all('Content-Length') or []
                    if self.headers.get_all('Transfer-Encoding'):
                        self.reply(400, {'error': 'chunked_upload_not_supported'})
                        return
                    if not lengths:
                        self.reply(411, {'error': 'content_length_required'})
                        return
                    if (len(lengths) != 1 or len(lengths[0]) > 20
                            or not re.match(r'^[0-9]+$', lengths[0])):
                        self.reply(400, {'error': 'invalid_content_length'})
                        return
                    length = int(lengths[0])
                    if length > MAX_MODEL_BYTES:
                        self.reply(413, {'error': 'model_exceeds_512_MiB'})
                        return
                    if length < 1:
                        self.reply(400, {'error': 'empty_model'})
                        return
                    mode = self.headers.get('X-Model-Mode', '')
                    name = self.headers.get('X-Filename', '')
                    digest = self.headers.get('X-SHA256', '')
                    if (mode not in MODES or not _DIGEST.match(digest)
                            or not name.lower().endswith('.onnx') or len(name) > 200
                            or name != name.strip() or any(c in name for c in '/\\:')
                            or any(ord(c) < 32 for c in name)):
                        self.reply(400, {'error': 'invalid_model_metadata'})
                        return
                    with receiver._lock:
                        if not receiver._allowed(token):
                            self.reply(403, {'error': 'admission_closed_or_unauthorized'})
                            return
                        if receiver._uploading:
                            self.reply(409, {'error': 'upload_already_in_progress'})
                            return
                        receiver._uploading = True
                    job_id = uuid.uuid4().hex
                    partial = receiver.inbox / (job_id + '.part')
                    target = receiver.inbox / (job_id + '.onnx')
                    committed = False
                    response_code, response_body = 202, {'job_id': job_id, 'state': 'queued'}
                    try:
                        hasher = hashlib.sha256()
                        remaining = length
                        with partial.open('xb') as output:
                            while remaining:
                                block = self.rfile.read(min(65536, remaining))
                                if not block:
                                    raise IOError('Incomplete upload')
                                output.write(block)
                                hasher.update(block)
                                remaining -= len(block)
                                self._body_read += len(block)
                            output.flush()
                            import os
                            os.fsync(output.fileno())
                        if not hmac.compare_digest(hasher.hexdigest(), digest.lower()):
                            raise _UploadRejected(422, 'sha256_mismatch')
                        with receiver._lock:
                            if not receiver._allowed(token):
                                raise _UploadRejected(403, 'admission_expired_during_upload')
                            partial.replace(target)
                            receiver._token = None
                            receiver._expires = 0.0
                            receiver._jobs[job_id] = {
                                'job_id': job_id, 'state': 'queued',
                                'message': '文件已接收，等待板端检查和转换',
                                'details': {}, 'name': name, 'mode': mode,
                                'bytes': length, 'sha256': digest.lower(),
                                '_token': token, '_updated': receiver._clock()}
                            committed = True
                    except _UploadRejected as error:
                        response_code, response_body = error.code, {'error': error.error}
                    except (OSError, ValueError):
                        response_code, response_body = 400, {'error': 'upload_interrupted'}
                    finally:
                        if not committed and partial.exists():
                            try:
                                partial.unlink()
                            except OSError:
                                pass
                        with receiver._lock:
                            receiver._uploading = False
                    if committed:
                        receiver._event({'event': 'received', 'job_id': job_id,
                                         'mode': mode, 'bytes': length})
                        try:
                            receiver.on_received(target, mode, job_id)
                        except Exception:
                            receiver.update_job(job_id, 'failed', '板端无法排入转换队列')
                    try:
                        self.reply(response_code, response_body)
                    except OSError:
                        pass

            http_server = _ThreadedServer((self.host, self.port), Handler)
            udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                udp.bind((self.host, self.discovery_port))
                udp.settimeout(.2)
            except Exception:
                http_server.server_close()
                udp.close()
                raise
            self._http, self._udp = http_server, udp
            self.port = http_server.server_address[1]
            self.discovery_port = udp.getsockname()[1]
            self._threads = [threading.Thread(target=http_server.serve_forever,
                                              kwargs={'poll_interval': .1}),
                             threading.Thread(target=self._discover_loop)]
            for thread in self._threads:
                thread.daemon = True
                thread.start()
        return self.port

    def _allowed(self, token):
        return (not self._stop.is_set() and self._token is not None
                and self._clock() < self._expires
                and isinstance(token, str)
                and re.match(r'^[A-Za-z0-9_-]{32,128}$', token) is not None
                and hmac.compare_digest(self._token, token))

    def _event(self, value):
        if self.on_event is not None:
            try:
                self.on_event(dict(value))
            except Exception:
                pass

    def status(self):
        with self._lock:
            remaining = max(0, int(math.ceil(self._expires - self._clock())))
            return {'enabled': bool(self._token and remaining and not self._stop.is_set()),
                    'expires_in': remaining, 'port': self.port,
                    'discovery_port': self.discovery_port,
                    'uploading': self._uploading, 'receiver_id': self.receiver_id}

    def allow(self, seconds=300):
        if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or not 1 <= seconds <= 600:
            raise ValueError('Admission window must be 1..600 seconds')
        with self._lock:
            if self._http is None or self._stop.is_set():
                raise RuntimeError('Start receiver before allowing uploads')
            if self._uploading or any(j['state'] not in _TERMINAL for j in self._jobs.values()):
                raise RuntimeError('Previous model is still being received or prepared')
            self._token = secrets.token_urlsafe(32)
            self._expires = self._clock() + seconds
            self._jobs = {key: job for key, job in self._jobs.items()
                          if self._clock() - job['_updated'] < 3600}
            status = self.status()
        self._event(dict(status, event='allowed'))
        return status

    def _discover_loop(self):
        while not self._stop.is_set():
            try:
                payload, sender = self._udp.recvfrom(2048)
                request = json.loads(payload.decode('utf-8'))
                if (not isinstance(request, dict) or request.get('magic') != MAGIC
                        or request.get('action') != 'discover'
                        or not isinstance(request.get('nonce'), str)
                        or not re.match(r'^[0-9a-f]{32}$', request['nonce'])):
                    continue
                with self._lock:
                    if not self._token or not self._allowed(self._token) or self._uploading:
                        continue
                    reply = {'magic': MAGIC, 'nonce': request['nonce'], 'port': self.port,
                             'receiver_id': self.receiver_id, 'token': self._token,
                             'expires_in': self.status()['expires_in'], 'version': 1}
                self._udp.sendto(_json_bytes(reply), sender)
            except socket.timeout:
                continue
            except (OSError, ValueError, TypeError):
                if self._stop.is_set():
                    break

    def _get_job(self, job_id, token):
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return 404, {'error': 'unknown_job'}
            if (not isinstance(token, str) or not re.match(r'^[A-Za-z0-9_-]{32,128}$', token)
                    or not hmac.compare_digest(job['_token'], token)
                    or self._clock() - job['_updated'] > 3600):
                return 403, {'error': 'job_unauthorized_or_expired'}
            return 200, json.loads(_json_bytes({key: value for key, value in job.items()
                                               if not key.startswith('_')}).decode('utf-8'))

    def update_job(self, job_id, state, message='', details=None):
        if state not in _JOB_STATES:
            raise ValueError('Unsupported model job state')
        details = {} if details is None else details
        payload = _json_bytes({'message': str(message), 'details': details})
        if len(payload) > 16384:
            raise ValueError('Job status metadata exceeds 16 KiB')
        clean = json.loads(payload.decode('utf-8'))
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or (job['state'] in _TERMINAL and job['state'] != state):
                return False
            # Callers receive no token. Also redact it if accidentally echoed.
            text = _json_bytes(clean).decode('utf-8').replace(job['_token'], '[redacted]')
            clean = json.loads(text)
            job.update(state=state, message=clean['message'], details=clean['details'],
                       _updated=self._clock())
        self._event({'event': 'job', 'job_id': job_id, 'state': state,
                     'message': clean['message']})
        return True

    def close(self):
        with self._lock:
            if self._stop.is_set():
                return
            self._stop.set()
            self._token = None
            self._expires = 0
            connections = list(self._connections)
            http_server, udp = self._http, self._udp
        for connection in connections:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if udp is not None:
            udp.close()
        if http_server is not None:
            http_server.shutdown()
            http_server.server_close()
        for thread in self._threads:
            if thread is not threading.current_thread():
                thread.join(timeout=1)


def discover_receivers(timeout=2.0, address=None, discovery_port=DISCOVERY_PORT):
    """Broadcast discovery, or UDP unicast to a manually entered LAN IPv4."""
    timeout = float(timeout)
    if not math.isfinite(timeout) or not 0 < timeout <= 10:
        raise ValueError('Discovery timeout must be 0..10 seconds')
    destination = _lan_ip(address) if address else '255.255.255.255'
    nonce = uuid.uuid4().hex
    request = _json_bytes({'magic': MAGIC, 'action': 'discover', 'nonce': nonce})
    found = {}
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp:
        udp.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        udp.bind(('0.0.0.0', 0))
        udp.sendto(request, (destination, int(discovery_port)))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            udp.settimeout(max(.001, deadline - time.monotonic()))
            try:
                payload, source = udp.recvfrom(4096)
                result = json.loads(payload.decode('utf-8'))
                if (not isinstance(result, dict) or result.get('magic') != MAGIC
                        or result.get('nonce') != nonce or result.get('version') != 1
                        or not isinstance(result.get('port'), int)
                        or not 1 <= result['port'] <= 65535
                        or not isinstance(result.get('token'), str)
                        or not re.match(r'^[A-Za-z0-9_-]{32,128}$', result['token'])
                        or not isinstance(result.get('receiver_id'), str)
                        or not _JOB_ID.match(result['receiver_id'])):
                    continue
                ip = _lan_ip(source[0])
                if address and ip != destination:
                    continue
                result['host'] = ip
                found[result['receiver_id']] = result
            except socket.timeout:
                break
            except (ValueError, TypeError, KeyError):
                continue
    return list(found.values())


def _endpoint(endpoint):
    if not isinstance(endpoint, dict):
        raise ValueError('Use a receiver returned by LAN discovery')
    host = _lan_ip(endpoint.get('host'))
    port = int(endpoint.get('port', 0))
    token = endpoint.get('token', '')
    if (not 1 <= port <= 65535 or not isinstance(token, str)
            or not re.match(r'^[A-Za-z0-9_-]{32,128}$', token)):
        raise ValueError('Invalid receiver admission')
    return host, port, token


def _response(connection):
    reply = connection.getresponse()
    payload = reply.read(65537)
    if len(payload) > 65536:
        raise IOError('Receiver returned an oversized response')
    try:
        value = json.loads(payload.decode('utf-8'))
    except (ValueError, UnicodeError):
        raise IOError('Receiver returned an invalid response')
    if reply.status >= 400:
        raise IOError('板端拒绝请求: ' + str(value.get('error', reply.status)))
    return value


def send_model(endpoint, path, mode='legacy_features', on_progress=None):
    """Transport only. The GUI validates the ONNX contract before calling."""
    host, port, token = _endpoint(endpoint)
    path = Path(path)
    if path.suffix.lower() != '.onnx' or not path.is_file() or mode not in MODES:
        raise ValueError('请选择 ONNX 文件和有效输入模式')
    length = path.stat().st_size
    if not 1 <= length <= MAX_MODEL_BYTES:
        raise ValueError('ONNX 文件大小必须在 1 字节到 512 MiB 之间')
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    connection = http.client.HTTPConnection(host, port, timeout=30)
    try:
        connection.putrequest('POST', '/v1/upload')
        # The actual storage name is receiver-generated. Avoid HTTP header
        # encoding problems with arbitrary Windows/Chinese source filenames.
        for key, value in (('Authorization', 'Bearer ' + token),
                           ('Content-Length', str(length)), ('X-Filename', 'model.onnx'),
                           ('X-Model-Mode', mode), ('X-SHA256', digest.hexdigest())):
            connection.putheader(key, value)
        connection.endheaders()
        sent = 0
        with path.open('rb') as source:
            while sent < length:
                block = source.read(min(65536, length - sent))
                if not block:
                    raise IOError('Model changed during transfer; please retry')
                connection.send(block)
                sent += len(block)
                if on_progress is not None:
                    on_progress(sent, length)
        return _response(connection)
    finally:
        connection.close()


def get_job(endpoint, job_id):
    host, port, token = _endpoint(endpoint)
    if not isinstance(job_id, str) or not _JOB_ID.match(job_id):
        raise ValueError('Invalid job identifier')
    connection = http.client.HTTPConnection(host, port, timeout=10)
    try:
        connection.request('GET', '/v1/jobs/' + job_id,
                           headers={'Authorization': 'Bearer ' + token})
        return _response(connection)
    finally:
        connection.close()
