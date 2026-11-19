"""Versioned side detector used by the real live console, not the replay app."""
from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import uuid

import numpy as np
import yaml

from trt_backend import create_backend

ROOT = Path(__file__).resolve().parent
DEFAULT_SAVED_APP = Path.home() / 'sorter-update-1114' / 'jetson-app'
BUILD_FLAGS = ('--fp16', '--workspace=1024', '--buildOnly')
SHAPES = {'ban_yuan_zhu': '半圆柱', 'chang_fang_ti': '长方体', 'liu_leng_zhu': '六棱柱',
          'san_leng_zhu': '三棱柱', 'san_leng_zhui': '三棱锥', 'si_leng_zhui': '四棱锥',
          'wu_leng_zhu': '五棱柱', 'yuan_zhu': '圆柱', 'yuan_zhui': '圆锥', 'zheng_fang_ti': '正方体'}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_names(names):
    if (not isinstance(names, (list, tuple)) or not 1 <= len(names) <= 1000
            or any(not isinstance(n, str) or not n.strip() or '\n' in n or '\r' in n
                   or n.startswith('reserved_') for n in names) or len(set(names)) != len(names)):
        raise ValueError('类别清单必须是有序、非空、无重复的真实类别')
    return list(names)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('w', encoding='utf8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass(frozen=True)
class SideCandidate:
    manifest_path: Path
    manifest: dict


class SideModelStore:
    def __init__(self, app_root=ROOT):
        self.root = Path(app_root)
        self.versions = self.root / 'models' / 'live_side_versions'
        self.active_path = self.root / 'data' / 'live_side_model.json'

    def _new_version(self):
        stage = self.versions / uuid.uuid4().hex
        stage.mkdir(parents=True)
        return stage

    def _read_manifest(self, path):
        path = Path(path)
        versions = self.versions.resolve()
        if path.is_symlink() or versions not in path.resolve().parents:
            raise ValueError('模型候选必须位于本程序的版本目录')
        record = json.loads(path.read_text(encoding='utf8'))
        names = validate_names(record['class_names'])
        if (record['schema'] != 1 or record['input_shape'] != [1, 3, 320, 320]
                or record['output_shape'] != [1, 4 + len(names), 2100]
                or record['preprocess'] not in ('legacy_features', 'raw_rgb')
                or record['engine'] != 'shape_fp16.engine'):
            raise ValueError('模型尺寸、类别数或输入模式不兼容')
        engine = path.parent / record['engine']
        if engine.is_symlink() or sha256(engine) != record['engine_sha256']:
            raise ValueError('Engine 校验失败')
        return record

    def read_active(self):
        if not self.active_path.exists():
            return None
        pointer = json.loads(self.active_path.read_text(encoding='utf8'))
        version = pointer.get('version')
        if not isinstance(version, str) or len(version) != 32 or any(c not in '0123456789abcdef' for c in version):
            raise ValueError('活动模型版本标识无效')
        candidate = self.versions / pointer['version'] / 'manifest.json'
        record = self._read_manifest(candidate)
        if pointer['version'] != candidate.parent.name or pointer['engine_sha256'] != record['engine_sha256']:
            raise ValueError('活动模型指针校验失败')
        record['engine_path'] = str(candidate.parent / record['engine'])
        record['version'] = candidate.parent.name
        return record

    def _smoke_engine(self, engine, names):
        backend = create_backend(engine, 'yolo_side', side_class_count=len(names))
        try:
            output = backend.infer(np.zeros((1, 3, 320, 320), np.float32))
            if not isinstance(output, np.ndarray) or output.shape != (1, 4 + len(names), 2100) or not np.isfinite(output).all():
                raise ValueError('Engine 试推理失败：输出尺寸或数值异常')
        finally:
            backend.close()

    def _finish(self, stage, names, mode, source, progress, fingerprint=None):
        names = validate_names(names)
        if mode not in ('legacy_features', 'raw_rgb'):
            raise ValueError('请选择正确的训练输入模式')
        progress('正在校验 Engine 并试推理…')
        engine_sha = sha256(stage / 'shape_fp16.engine')
        self._smoke_engine(stage / 'shape_fp16.engine', names)
        if sha256(stage / 'shape_fp16.engine') != engine_sha:
            raise ValueError('Engine 在试推理期间发生变化，候选未发布')
        record = {'schema': 1, 'class_names': names, 'input_shape': [1, 3, 320, 320],
                  'output_shape': [1, 4 + len(names), 2100], 'preprocess': mode,
                  'engine': 'shape_fp16.engine', 'engine_sha256': engine_sha,
                  'source': source}
        if fingerprint is not None:
            record['conversion_fingerprint'] = fingerprint
        path = stage / 'manifest.json'
        atomic_json(path, record)
        return SideCandidate(path, record)

    def prepare_existing(self, saved_app=DEFAULT_SAVED_APP, mode='legacy_features', progress=lambda message: None):
        saved_app = Path(saved_app)
        version = json.loads((saved_app / 'data' / 'active_model.json').read_text(encoding='utf8'))['bundle']
        if not isinstance(version, str) or len(version) != 32 or any(c not in '0123456789abcdef' for c in version):
            raise ValueError('测试程序的模型版本标识无效')
        bundle = saved_app / 'models' / 'imports' / version
        info = json.loads((bundle / 'manifest.json').read_text(encoding='utf8'))
        names = validate_names(info['names'])
        if info['input_size'] != 320 or info['output_shape'] != [1, 4 + len(names), 2100]:
            raise ValueError('正式程序目前需要单视图 320×320 raw 模型，请重新导出对应尺寸')
        if sha256(bundle / 'model.onnx') != info['sha256']:
            raise ValueError('来源 ONNX 校验不一致')
        source_engine = bundle / 'model_fp16.engine'
        if not source_engine.is_file():
            raise ValueError('来源模型尚无 Jetson Engine，请先完成构建')
        stage = self._new_version()
        progress('正在复用已构建 Engine，不会重新编译…')
        shutil.copy2(str(source_engine), str(stage / 'shape_fp16.engine'))
        return self._finish(stage, names, mode, {'saved_app': str(saved_app),
                            'onnx_sha256': info['sha256'], 'bundle': version}, progress)

    def inspect_onnx(self, source):
        """Inspect metadata without ONNX, TensorRT, CUDA or camera initialization."""
        from onnx_contract import inspect_yolo_onnx
        return inspect_yolo_onnx(source)

    def conversion_fingerprint(self):
        """Identify the native builder/runtime/hardware, never a guessed version."""
        exe = Path(shutil.which('trtexec') or '/usr/src/tensorrt/bin/trtexec')
        try:
            trt = importlib.import_module('tensorrt')
            cuda = ctypes.CDLL('libcudart.so')
            values = {}
            for name in ('cudaRuntimeGetVersion', 'cudaDriverGetVersion'):
                function = getattr(cuda, name)
                function.argtypes, function.restype = [ctypes.POINTER(ctypes.c_int)], ctypes.c_int
                value = ctypes.c_int()
                if function(ctypes.byref(value)) != 0 or value.value <= 0:
                    raise ValueError('Cannot query ' + name)
                values[name] = value.value
            attribute = cuda.cudaDeviceGetAttribute
            attribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
            attribute.restype = ctypes.c_int
            capability = []
            for code in (75, 76):  # cudaDevAttrComputeCapabilityMajor / Minor.
                value = ctypes.c_int()
                if attribute(ctypes.byref(value), code, 0) != 0:
                    raise ValueError('Cannot query CUDA device capability')
                capability.append(value.value)
            def text(path):
                file = Path(path)
                return file.read_text(encoding='utf8').strip('\x00\n ') if file.is_file() else ''
            return {'schema': 1, 'tensorrt': str(trt.__version__),
                    'cuda_runtime': values['cudaRuntimeGetVersion'], 'cuda_driver': values['cudaDriverGetVersion'],
                    'compute_capability': capability, 'hardware': text('/proc/device-tree/model'),
                    'compatible': text('/proc/device-tree/compatible'), 'l4t': text('/etc/nv_tegra_release'),
                    'machine': platform.machine(), 'platform': platform.platform(),
                    'trtexec_path': str(exe.resolve()), 'trtexec_sha256': sha256(exe), 'flags': list(BUILD_FLAGS)}
        except (ImportError, OSError, AttributeError, ValueError) as error:
            raise ValueError('无法确认本机 TensorRT/CUDA 构建环境：' + str(error)) from error

    @contextmanager
    def _conversion_lock(self):
        """An OS-held lock is released on process exit, without deleting its inode."""
        self.versions.mkdir(parents=True, exist_ok=True)
        with (self.versions / '.conversion.lock').open('a+b') as stream:
            if stream.seek(0, os.SEEK_END) == 0:
                stream.write(b'\0')
                stream.flush()
            stream.seek(0)
            try:
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (OSError, IOError) as error:
                raise ValueError('已有模型转换正在进行，请等待完成后重试') from error
            try:
                yield
            finally:
                stream.seek(0)
                if os.name == 'nt':
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def _running_trtexec(self, proc=Path('/proc'), uid=None):
        if not proc.is_dir() or (uid is None and not hasattr(os, 'geteuid')):
            return []
        uid = os.geteuid() if uid is None else uid
        running = []
        for folder in proc.iterdir():
            if not folder.name.isdigit() or int(folder.name) == os.getpid():
                continue
            try:
                if folder.stat().st_uid != uid:
                    continue
                argv = (folder / 'cmdline').read_bytes().split(b'\0')
                if argv and Path(os.fsdecode(argv[0])).name == 'trtexec':
                    running.append(int(folder.name))
            except (OSError, IOError):
                continue
        return running

    def find_cached(self, contract, mode, fingerprint, progress=lambda message: None):
        paths = sorted(self.versions.glob('*/manifest.json'), key=lambda path: path.stat().st_mtime, reverse=True)
        for path in paths:
            try:
                record = self._read_manifest(path)
                if (record.get('source', {}).get('onnx_sha256') != contract['sha256']
                        or record['class_names'] != list(contract['names']) or record['preprocess'] != mode):
                    continue
                if ('conversion_fingerprint' in record and record['conversion_fingerprint'] != fingerprint):
                    continue
                onnx_path = path.parent / 'model.onnx'
                if onnx_path.exists() and (onnx_path.is_symlink() or sha256(onnx_path) != contract['sha256']):
                    raise ValueError('缓存 ONNX 内容校验不一致')
                progress('正在验证已构建 Engine 缓存：' + path.parent.name)
                self._smoke_engine(path.parent / record['engine'], contract['names'])
                if sha256(path.parent / record['engine']) != record['engine_sha256']:
                    raise ValueError('缓存 Engine 在验证期间发生变化')
                progress('已复用本机验证通过的 Engine，跳过 trtexec 编译。')
                return SideCandidate(path, record)
            except Exception as error:
                progress('跳过无效 Engine 缓存 %s：%s' % (path.parent.name, error))
        return None

    def prepare_onnx(self, source, mode='legacy_features', progress=lambda message: None):
        if mode not in ('legacy_features', 'raw_rgb'):
            raise ValueError('请选择正确的训练输入模式')
        source = Path(source)
        contract = self.inspect_onnx(source)
        names = validate_names(contract['names'])
        if (contract['input_shape'] != [1, 3, 320, 320] or contract['output_shape'] != [1, 4 + len(names), 2100]
                or sha256(source) != contract['sha256']):
            raise ValueError('ONNX 内容或输入输出在检查后发生变化')
        with self._conversion_lock():
            fingerprint = self.conversion_fingerprint()
            cached = self.find_cached(contract, mode, fingerprint, progress)
            if cached is not None:
                return cached
            running = self._running_trtexec()
            if running:
                raise ValueError('已有本用户 trtexec 正在构建（PID %s），请等待完成；不会终止该进程' % running)
            stage = self._new_version()
            shutil.copy2(str(source), str(stage / 'model.onnx'))
            if sha256(stage / 'model.onnx') != contract['sha256']:
                raise ValueError('ONNX 在复制期间发生变化，未启动构建')
            if self._running_trtexec():
                raise ValueError('已有本用户 trtexec 在复制期间开始构建，请等待完成后重试')
            progress('未找到同环境有效缓存，正在生成 FP16 Engine，最多等待 30 分钟…')
            command = [fingerprint.get('trtexec_path') or shutil.which('trtexec') or '/usr/src/tensorrt/bin/trtexec',
                       '--onnx=' + str(stage / 'model.onnx'), '--saveEngine=' + str(stage / 'shape_fp16.engine')]
            with (stage / 'build.log').open('w', encoding='utf8') as log:
                result = subprocess.run(command + list(BUILD_FLAGS), stdout=log, stderr=subprocess.STDOUT, timeout=1800)
            if result.returncode:
                text = (stage / 'build.log').read_text(encoding='utf8', errors='replace')
                raise ValueError('构建失败，日志：%s\n%s' % (stage / 'build.log', text[:600] + '\n' + text[-900:]))
            if sha256(stage / 'model.onnx') != contract['sha256']:
                raise ValueError('ONNX 在构建期间发生变化，候选未发布')
            return self._finish(stage, names, mode, {'onnx_sha256': contract['sha256'],
                'source_name': source.name, 'onnx_contract': contract}, progress, fingerprint=fingerprint)

    def rollback_previous(self, progress=lambda message: None):
        """Validate and natively smoke the previous release before atomic activation."""
        if not self.active_path.is_file():
            raise ValueError('当前没有可回退的活动模型')
        active_before = self.active_path.read_bytes()
        pointer = json.loads(active_before.decode('utf8'))
        previous = pointer.get('previous')
        version = previous.get('version') if isinstance(previous, dict) else None
        if not isinstance(version, str) or len(version) != 32 or any(c not in '0123456789abcdef' for c in version):
            raise ValueError('没有有效的上一版模型可回退')
        path = self.versions / version / 'manifest.json'
        record = self._read_manifest(path)
        if previous.get('engine_sha256') != record['engine_sha256']:
            raise ValueError('上一版模型指针校验失败')
        progress('正在校验上一版 Engine 并试推理…')
        self._smoke_engine(path.parent / record['engine'], record['class_names'])
        if sha256(path.parent / record['engine']) != record['engine_sha256']:
            raise ValueError('上一版 Engine 在校验期间发生变化')
        if self.active_path.read_bytes() != active_before:
            raise ValueError('活动模型在回退校验期间发生变化，请重试')
        candidate = SideCandidate(path, record)
        self.activate(candidate)
        return candidate

    def activate(self, candidate):
        record = self._read_manifest(candidate.manifest_path)
        if record != candidate.manifest:
            raise ValueError('模型候选在试推理后发生变化，请重新准备')
        pointer = {'version': candidate.manifest_path.parent.name, 'engine_sha256': record['engine_sha256']}
        if self.active_path.exists():
            current = json.loads(self.active_path.read_text(encoding='utf8'))
            if (isinstance(current, dict) and current.get('version') == pointer['version']
                    and current.get('engine_sha256') == pointer['engine_sha256']):
                self.read_active()  # Validate the existing pointer before treating this as a no-op.
                return
            # Keep a restore point without touching original 24-class weights or settings.
            pointer['previous'] = current
        atomic_json(self.active_path, pointer)


class LiveSideProfile:
    """Snapshot one active version so a running round cannot silently switch names."""
    def __init__(self, root, base_profile, record):
        self.root = Path(root)
        self.base = base_profile
        self.record = record
        self.CLASS_NAMES = tuple(record['class_names'])
        self.SHAPES = dict(getattr(base_profile, 'SHAPES', {}), **SHAPES)

    def make_recognizer(self):
        from legacy_vision import LegacyRecognizer
        config = yaml.safe_load((self.root / 'legacy_vision.yaml').read_text(encoding='utf8'))
        config.setdefault('surface_mask', {})['enabled'] = False
        detector = create_backend(self.record['engine_path'], 'yolo_side', side_class_count=len(self.CLASS_NAMES))
        try:
            return LegacyRecognizer(detector, None, self.CLASS_NAMES, config,
                                    expected_count=len(self.CLASS_NAMES), per_view_features=True,
                                    per_view_input_mode=self.record['preprocess'], capture_color_sample=True)
        except Exception:
            detector.close()
            raise
