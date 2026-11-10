"""TensorRT 8.5 backend using only the CUDA Runtime through ctypes.

Importing this module is safe on machines without TensorRT or CUDA.  Native
dependencies are loaded only by :func:`create_backend`; tests can inject the
same narrow TensorRT and CUDA interfaces directly into :class:`TensorRTBackend`.
"""

import ctypes
import hashlib
import importlib
import os
import threading
from pathlib import Path

import numpy as np


CUDA_MEMCPY_HOST_TO_DEVICE = 1
CUDA_MEMCPY_DEVICE_TO_HOST = 2
CUDA_STREAM_NON_BLOCKING = 1

_INPUT_SHAPE = (1, 3, 320, 320)
_YOLO_OUTPUT_SHAPE = (1, 27, 2100)
_YOLO_LEGACY_INPUT_SHAPE = (1, 3, 640, 640)
_YOLO_LEGACY_OUTPUT_SHAPE = (1, 30, 8400)
_YOLO_WIN8_OUTPUT_SHAPE = (1, 12, 8400)
_MASK_OUTPUT_SHAPE = (1, 1, 320, 320)


class BackendUnavailable(RuntimeError):
    """A native dependency required to construct the backend is unavailable."""


class CudaError(RuntimeError):
    """A CUDA Runtime operation returned a non-zero cudaError_t."""


class EngineContractError(RuntimeError):
    """A serialized engine does not match the fixed application contract."""


class TensorRTExecutionError(RuntimeError):
    """TensorRT refused to enqueue an inference."""


class ThreadOwnershipError(RuntimeError):
    """A backend was used from a thread other than its creating thread."""


class CudaRuntime:
    """Precisely typed subset of the CUDA Runtime API used by the backend."""

    def __init__(self, library):
        self._library = library
        self._declare_signatures()

    def _declare_signatures(self):
        library = self._library
        library.cudaSetDevice.argtypes = [ctypes.c_int]
        library.cudaSetDevice.restype = ctypes.c_int
        library.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        library.cudaMalloc.restype = ctypes.c_int
        library.cudaFree.argtypes = [ctypes.c_void_p]
        library.cudaFree.restype = ctypes.c_int
        library.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        library.cudaMemcpyAsync.restype = ctypes.c_int
        library.cudaStreamCreateWithFlags.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint,
        ]
        library.cudaStreamCreateWithFlags.restype = ctypes.c_int
        library.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        library.cudaStreamSynchronize.restype = ctypes.c_int
        library.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        library.cudaStreamDestroy.restype = ctypes.c_int
        library.cudaGetErrorString.argtypes = [ctypes.c_int]
        library.cudaGetErrorString.restype = ctypes.c_char_p

    def _check(self, code, operation):
        numeric_code = int(code)
        if numeric_code == 0:
            return
        raw_message = self._library.cudaGetErrorString(numeric_code)
        if raw_message:
            message = raw_message.decode("utf-8", "replace")
        else:
            message = "unknown CUDA error"
        raise CudaError(
            "%s failed (%d): %s" % (operation, numeric_code, message)
        )

    def set_device(self, device):
        self._check(self._library.cudaSetDevice(ctypes.c_int(device)), "cudaSetDevice")

    def malloc(self, nbytes):
        pointer = ctypes.c_void_p()
        self._check(
            self._library.cudaMalloc(ctypes.byref(pointer), ctypes.c_size_t(nbytes)),
            "cudaMalloc",
        )
        if pointer.value is None:
            raise CudaError("cudaMalloc returned a null device pointer")
        return int(pointer.value)

    def free(self, pointer):
        self._check(self._library.cudaFree(ctypes.c_void_p(pointer)), "cudaFree")

    def memcpy_async(self, destination, source, nbytes, kind, stream):
        self._check(
            self._library.cudaMemcpyAsync(
                ctypes.c_void_p(destination),
                ctypes.c_void_p(source),
                ctypes.c_size_t(nbytes),
                ctypes.c_int(kind),
                ctypes.c_void_p(stream),
            ),
            "cudaMemcpyAsync",
        )

    def create_stream(self, flags=CUDA_STREAM_NON_BLOCKING):
        stream = ctypes.c_void_p()
        self._check(
            self._library.cudaStreamCreateWithFlags(
                ctypes.byref(stream), ctypes.c_uint(flags)
            ),
            "cudaStreamCreateWithFlags",
        )
        if stream.value is None:
            raise CudaError("cudaStreamCreateWithFlags returned a null stream")
        return int(stream.value)

    def synchronize(self, stream):
        self._check(
            self._library.cudaStreamSynchronize(ctypes.c_void_p(stream)),
            "cudaStreamSynchronize",
        )

    def destroy_stream(self, stream):
        self._check(
            self._library.cudaStreamDestroy(ctypes.c_void_p(stream)),
            "cudaStreamDestroy",
        )


_cuda_runtime = None
_cuda_runtime_lock = threading.Lock()


def _load_cuda_runtime():
    global _cuda_runtime
    with _cuda_runtime_lock:
        if _cuda_runtime is not None:
            return _cuda_runtime
        mode = getattr(os, "RTLD_NOW", 0) | getattr(os, "RTLD_GLOBAL", 0)
        try:
            library = ctypes.CDLL("libcudart.so", mode=mode)
        except (OSError, AttributeError) as exc:
            raise BackendUnavailable(
                "CUDA Runtime library 'libcudart.so' is unavailable: %s" % exc
            ) from exc
        try:
            _cuda_runtime = CudaRuntime(library)
        except AttributeError as exc:
            raise BackendUnavailable(
                "CUDA Runtime library is missing a required symbol: %s" % exc
            ) from exc
        return _cuda_runtime


def create_backend(engine_path, model_kind, device=0):
    """Load native dependencies and create a backend on the calling thread."""

    try:
        trt_module = importlib.import_module("tensorrt")
    except (ImportError, OSError) as exc:
        raise BackendUnavailable("TensorRT Python bindings are unavailable: %s" % exc) from exc
    cuda_runtime = _load_cuda_runtime()
    try:
        engine_bytes = Path(engine_path).read_bytes()
    except OSError as exc:
        raise BackendUnavailable("cannot read TensorRT engine '%s': %s" % (engine_path, exc)) from exc
    provenance = {
        'engine_path': str(Path(engine_path).resolve()),
        'engine_sha256': hashlib.sha256(engine_bytes).hexdigest(),
        'model_kind': model_kind,
    }
    backend = TensorRTBackend(
        engine_bytes,
        model_kind,
        trt_module,
        cuda_runtime,
        device=device,
    )
    backend.provenance = provenance
    return backend


def _shape_nbytes(shape, dtype):
    if not shape or any(not isinstance(dimension, (int, np.integer)) or dimension <= 0 for dimension in shape):
        raise EngineContractError("cannot allocate unresolved shape %r" % (shape,))
    return int(np.prod(shape)) * np.dtype(dtype).itemsize


class TensorRTBackend:
    """One synchronous TensorRT engine owned by one worker thread."""

    def __init__(self, engine_bytes, model_kind, trt_module, cuda_runtime, device=0):
        if model_kind not in ("yolo", "yolo_legacy", "yolo_win8", "tray", "mask"):
            raise ValueError("model_kind must be 'yolo', 'yolo_legacy', 'yolo_win8', 'tray', or 'mask'")
        if int(device) != 0:
            raise ValueError("this backend is fixed to CUDA device 0")
        self.model_kind = model_kind
        self._trt = trt_module
        self._cuda = cuda_runtime
        self._device = int(device)
        self._owner_thread = threading.get_ident()
        self._closed = False

        self.logger = None
        self.runtime = None
        self.engine = None
        self.context = None
        self.stream = None
        self.input_device_pointer = None
        self.output_device_pointer = None
        self._host_output = None
        self.input_index = None
        self.output_index = None
        self.input_shape = None
        self.output_shape = None
        self.input_dtype = None
        self.output_dtype = None
        self.input_nbytes = None
        self.output_nbytes = None
        self._bindings = None

        try:
            self._cuda.set_device(self._device)
            self.stream = self._cuda.create_stream(CUDA_STREAM_NON_BLOCKING)
            self.logger = self._trt.Logger(self._trt.Logger.ERROR)
            self.runtime = self._trt.Runtime(self.logger)
            if self.runtime is None:
                raise EngineContractError("TensorRT runtime creation failed")
            self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
            if self.engine is None:
                raise EngineContractError("TensorRT engine deserialize failed")
            self.context = self.engine.create_execution_context()
            if self.context is None:
                raise EngineContractError("TensorRT execution context creation failed")
            self._validate_engine_contract()
            if self.model_kind == "mask":
                self.output_shape = self._configure_dynamic_shape()
            self._allocate_buffers()
        except BaseException:
            self._cleanup(suppress_errors=True)
            self._closed = True
            raise

    def _check_owner(self):
        if threading.get_ident() != self._owner_thread:
            raise ThreadOwnershipError(
                "TensorRT backend must be used from its owner thread"
            )

    def _validate_engine_contract(self):
        engine = self.engine
        if bool(engine.has_implicit_batch_dimension):
            raise EngineContractError("implicit batch engines are not supported")
        if int(engine.num_bindings) != 2:
            raise EngineContractError(
                "expected exactly 2 bindings, found %d" % int(engine.num_bindings)
            )
        profile_count = int(getattr(engine, "num_optimization_profiles", 1))
        if profile_count != 1:
            raise EngineContractError(
                "expected exactly one optimization profile, found %d" % profile_count
            )

        expected_input_name = "input" if self.model_kind == "mask" else "images"
        input_index = int(engine.get_binding_index(expected_input_name))
        if input_index < 0 or input_index >= int(engine.num_bindings):
            raise EngineContractError(
                "expected input binding '%s'" % expected_input_name
            )
        if not engine.binding_is_input(input_index):
            raise EngineContractError(
                "binding '%s' is not an input" % expected_input_name
            )
        output_indices = [
            index
            for index in range(int(engine.num_bindings))
            if index != input_index and not engine.binding_is_input(index)
        ]
        if len(output_indices) != 1:
            raise EngineContractError("expected exactly one output binding")
        output_index = output_indices[0]

        for index in range(int(engine.num_bindings)):
            name = engine.get_binding_name(index)
            if int(engine.get_binding_index(name)) != index:
                raise EngineContractError("inconsistent binding index for '%s'" % name)

        input_shape = tuple(int(value) for value in engine.get_binding_shape(input_index))
        declared_output_shape = tuple(
            int(value) for value in engine.get_binding_shape(output_index)
        )
        if self.model_kind != "mask":
            if self.model_kind == "tray":
                expected_input_shape = _YOLO_LEGACY_INPUT_SHAPE
                expected_output_shape = (1, 5, 8400)
            elif self.model_kind == "yolo_legacy":
                expected_input_shape = _YOLO_LEGACY_INPUT_SHAPE
                expected_output_shape = _YOLO_LEGACY_OUTPUT_SHAPE
            elif self.model_kind == "yolo_win8":
                expected_input_shape = _YOLO_LEGACY_INPUT_SHAPE
                expected_output_shape = _YOLO_WIN8_OUTPUT_SHAPE
            else:
                expected_input_shape = _INPUT_SHAPE
                expected_output_shape = _YOLO_OUTPUT_SHAPE
            if input_shape != expected_input_shape:
                raise EngineContractError(
                    "unexpected YOLO input shape %r; expected %r"
                    % (input_shape, expected_input_shape)
                )
            if declared_output_shape != expected_output_shape:
                raise EngineContractError(
                    "unexpected YOLO output shape %r; expected %r"
                    % (declared_output_shape, expected_output_shape)
                )
            self.output_shape = expected_output_shape
        else:
            expected_input_shape = _INPUT_SHAPE
            if len(input_shape) != 4 or any(
                actual not in (-1, expected)
                for actual, expected in zip(input_shape, _INPUT_SHAPE)
            ):
                raise EngineContractError(
                    "unexpected mask input shape %r" % (input_shape,)
                )
            if len(declared_output_shape) != 4 or any(
                actual not in (-1, expected)
                for actual, expected in zip(declared_output_shape, _MASK_OUTPUT_SHAPE)
            ):
                raise EngineContractError(
                    "unexpected mask declared output shape %r"
                    % (declared_output_shape,)
                )
            try:
                profile_shapes = tuple(
                    tuple(int(value) for value in shape)
                    for shape in engine.get_profile_shape(0, input_index)
                )
            except (AttributeError, TypeError, ValueError) as exc:
                raise EngineContractError(
                    "cannot query mask optimization profile 0: %s" % exc
                ) from exc
            expected_profile_shapes = (
                _INPUT_SHAPE,
                _INPUT_SHAPE,
                _INPUT_SHAPE,
            )
            if profile_shapes != expected_profile_shapes:
                raise EngineContractError(
                    "mask optimization profile 0 min/opt/max must all be %r; got %r"
                    % (_INPUT_SHAPE, profile_shapes)
                )

        try:
            input_dtype = np.dtype(self._trt.nptype(engine.get_binding_dtype(input_index)))
            output_dtype = np.dtype(self._trt.nptype(engine.get_binding_dtype(output_index)))
        except (KeyError, TypeError, ValueError) as exc:
            raise EngineContractError("unsupported TensorRT binding dtype: %s" % exc) from exc
        if input_dtype.kind != "f" or output_dtype.kind != "f":
            raise EngineContractError("TensorRT input and output bindings must be floating point")

        self.input_index = input_index
        self.output_index = output_index
        self.input_shape = expected_input_shape
        self.input_dtype = input_dtype
        self.output_dtype = output_dtype

    def _configure_dynamic_shape(self):
        profile_ok = self.context.set_optimization_profile_async(0, int(self.stream))
        if profile_ok is False:
            raise EngineContractError("cannot select optimization profile 0")
        if not self.context.set_binding_shape(self.input_index, _INPUT_SHAPE):
            raise EngineContractError("mask input shape is outside profile 0")
        if not self.context.all_binding_shapes_specified:
            raise EngineContractError("not all dynamic binding shapes are specified")
        output_shape = tuple(
            int(value) for value in self.context.get_binding_shape(self.output_index)
        )
        if output_shape != _MASK_OUTPUT_SHAPE or any(value < 0 for value in output_shape):
            raise EngineContractError(
                "unexpected or unresolved mask output shape %r" % (output_shape,)
            )
        return output_shape

    def _allocate_buffers(self):
        self.input_nbytes = _shape_nbytes(self.input_shape, self.input_dtype)
        self.output_nbytes = _shape_nbytes(self.output_shape, self.output_dtype)
        self._host_output = np.empty(self.output_shape, dtype=self.output_dtype)
        self.input_device_pointer = self._cuda.malloc(self.input_nbytes)
        self.output_device_pointer = self._cuda.malloc(self.output_nbytes)
        self._bindings = [0] * int(self.engine.num_bindings)
        self._bindings[self.input_index] = int(self.input_device_pointer)
        self._bindings[self.output_index] = int(self.output_device_pointer)

    def infer(self, input_array):
        self._check_owner()
        if self._closed:
            raise RuntimeError("TensorRT backend is closed")
        if not isinstance(input_array, np.ndarray):
            raise TypeError("input must be a numpy ndarray")
        if tuple(input_array.shape) != self.input_shape:
            raise ValueError(
                "input shape %r does not match expected %r"
                % (tuple(input_array.shape), self.input_shape)
            )
        if input_array.dtype.kind not in "biuf":
            raise TypeError("input dtype must be numeric, got %s" % input_array.dtype)
        contiguous_input = np.ascontiguousarray(input_array, dtype=self.input_dtype)
        if contiguous_input.nbytes != self.input_nbytes:
            raise ValueError(
                "input nbytes %d does not match expected %d"
                % (contiguous_input.nbytes, self.input_nbytes)
            )

        if self.model_kind == "mask":
            output_shape = self._configure_dynamic_shape()
            if output_shape != self.output_shape:
                raise EngineContractError(
                    "mask output shape changed after buffer allocation"
                )

        self._cuda.memcpy_async(
            self.input_device_pointer,
            int(contiguous_input.ctypes.data),
            self.input_nbytes,
            CUDA_MEMCPY_HOST_TO_DEVICE,
            self.stream,
        )
        executed = self.context.execute_async_v2(
            bindings=self._bindings,
            stream_handle=int(self.stream),
        )
        if not executed:
            raise TensorRTExecutionError(
                "TensorRT execute_async_v2 returned False"
            )
        self._cuda.memcpy_async(
            int(self._host_output.ctypes.data),
            self.output_device_pointer,
            self.output_nbytes,
            CUDA_MEMCPY_DEVICE_TO_HOST,
            self.stream,
        )
        self._cuda.synchronize(self.stream)
        return self._host_output.copy()

    def close(self):
        self._check_owner()
        if self._closed:
            return
        self._closed = True
        errors = self._cleanup(suppress_errors=False)
        if errors:
            raise errors[0]

    def _cleanup(self, suppress_errors):
        errors = []

        def attempt(function, *args):
            try:
                function(*args)
            except BaseException as exc:
                errors.append(exc)

        if self.stream is not None:
            attempt(self._cuda.synchronize, self.stream)

        self.context = None
        self.engine = None
        self.runtime = None
        self._host_output = None
        self._bindings = None

        if self.output_device_pointer is not None:
            pointer = self.output_device_pointer
            self.output_device_pointer = None
            attempt(self._cuda.free, pointer)
        if self.input_device_pointer is not None:
            pointer = self.input_device_pointer
            self.input_device_pointer = None
            attempt(self._cuda.free, pointer)
        if self.stream is not None:
            stream = self.stream
            self.stream = None
            attempt(self._cuda.destroy_stream, stream)
        self.logger = None

        if suppress_errors:
            return []
        return errors


__all__ = [
    "BackendUnavailable",
    "CudaError",
    "CudaRuntime",
    "EngineContractError",
    "TensorRTBackend",
    "TensorRTExecutionError",
    "ThreadOwnershipError",
    "create_backend",
]
