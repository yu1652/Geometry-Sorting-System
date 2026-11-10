import ctypes
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class FakeFunction:
    def __init__(self, result=0, effect=None):
        self.result = result
        self.effect = effect
        self.argtypes = None
        self.restype = None
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        if self.effect is not None:
            self.effect(*args)
        return self.result


class FakeLibrary:
    def __init__(self, error_code=0):
        large_pointer = 2**40 + 123

        def set_pointer(out_pointer, _size):
            out_pointer._obj.value = large_pointer

        def set_stream(out_stream, _flags):
            out_stream._obj.value = 2**41 + 7

        self.cudaSetDevice = FakeFunction(error_code)
        self.cudaMalloc = FakeFunction(effect=set_pointer)
        self.cudaFree = FakeFunction()
        self.cudaMemcpyAsync = FakeFunction()
        self.cudaStreamCreateWithFlags = FakeFunction(effect=set_stream)
        self.cudaStreamSynchronize = FakeFunction()
        self.cudaStreamDestroy = FakeFunction()
        self.cudaGetErrorString = FakeFunction(result=b"injected CUDA failure")


class FakeContext:
    def __init__(
        self,
        events,
        output_shape,
        execute_result=True,
        profile_result=True,
        shape_result=True,
    ):
        self.events = events
        self.output_shape = output_shape
        self.execute_result = execute_result
        self.profile_result = profile_result
        self.shape_result = shape_result
        self.shapes_specified = True
        self.last_bindings = None

    def set_optimization_profile_async(self, profile, stream_handle):
        self.events.append(("profile", profile, stream_handle))
        return self.profile_result

    def set_binding_shape(self, index, shape):
        self.events.append(("set_shape", index, tuple(shape)))
        return self.shape_result

    @property
    def all_binding_shapes_specified(self):
        self.events.append(("all_shapes",))
        return self.shapes_specified

    def get_binding_shape(self, index):
        self.events.append(("output_shape", index))
        return self.output_shape

    def execute_async_v2(self, bindings, stream_handle):
        self.last_bindings = list(bindings)
        self.events.append(("execute", list(bindings), stream_handle))
        return self.execute_result


class FakeEngine:
    def __init__(
        self,
        events,
        kind="yolo",
        input_dtype="fp16",
        output_dtype="fp32",
        input_shape=None,
        output_shape=None,
        num_bindings=2,
        implicit=False,
        profiles=1,
        context=None,
        input_name=None,
        profile_shapes=None,
    ):
        self.events = events
        self.num_bindings = num_bindings
        self.has_implicit_batch_dimension = implicit
        self.num_optimization_profiles = profiles
        if input_shape is None:
            input_shape = (1, 3, 320, 320) if kind == "yolo" else (-1, 3, -1, -1)
        if output_shape is None:
            output_shape = (1, 27, 2100) if kind == "yolo" else (-1, 1, -1, -1)
        if input_name is None:
            input_name = "images" if kind == "yolo" else "input"
        self._names = ["predictions" if kind == "yolo" else "mask", input_name]
        self._shapes = [tuple(output_shape), tuple(input_shape)]
        self._dtypes = [output_dtype, input_dtype]
        if profile_shapes is None:
            profile_shapes = (
                (1, 3, 320, 320),
                (1, 3, 320, 320),
                (1, 3, 320, 320),
            )
        self.profile_shapes = tuple(tuple(shape) for shape in profile_shapes)
        resolved_output = (1, 27, 2100) if kind == "yolo" else (1, 1, 320, 320)
        self.context = context or FakeContext(events, resolved_output)

    def get_binding_index(self, name):
        try:
            return self._names.index(name)
        except ValueError:
            return -1

    def get_binding_name(self, index):
        return self._names[index]

    def get_binding_shape(self, index):
        return self._shapes[index]

    def get_binding_dtype(self, index):
        return self._dtypes[index]

    def binding_is_input(self, index):
        return index == 1

    def create_execution_context(self):
        self.events.append(("create_context",))
        return self.context

    def get_profile_shape(self, profile_index, binding_index):
        self.events.append(("get_profile_shape", profile_index, binding_index))
        return self.profile_shapes


class FakeLogger:
    ERROR = 3

    def __init__(self, severity):
        self.severity = severity


class FakeRuntime:
    def __init__(self, owner, logger):
        self.owner = owner
        self.logger = logger

    def deserialize_cuda_engine(self, engine_bytes):
        self.owner.deserialized = bytes(engine_bytes)
        return self.owner.engine


class FakeTrt:
    Logger = FakeLogger

    def __init__(self, engine):
        self.engine = engine
        self.runtime = None
        self.deserialized = None

    def Runtime(self, logger):
        self.runtime = FakeRuntime(self, logger)
        return self.runtime

    @staticmethod
    def nptype(dtype):
        return {"fp16": np.float16, "fp32": np.float32}[dtype]


class FakeCuda:
    def __init__(self, events, fail_at=None):
        self.events = events
        self.fail_at = fail_at
        self.allocations = []
        self.next_pointer = 2**40 + 100

    def _record(self, name, *args):
        self.events.append((name,) + args)
        if self.fail_at == name:
            from trt_backend import CudaError

            raise CudaError("%s injected failure" % name)

    def set_device(self, device):
        self._record("set_device", device)

    def create_stream(self, flags=1):
        self._record("create_stream", flags)
        return 2**42 + 9

    def malloc(self, nbytes):
        self._record("malloc", nbytes)
        pointer = self.next_pointer
        self.next_pointer += 17
        self.allocations.append(pointer)
        return pointer

    def free(self, pointer):
        self._record("free", pointer)

    def memcpy_async(self, destination, source, nbytes, kind, stream):
        self._record("memcpy", destination, source, nbytes, kind, stream)

    def synchronize(self, stream):
        self._record("sync", stream)

    def destroy_stream(self, stream):
        self._record("destroy_stream", stream)


def make_backend(kind="yolo", **engine_options):
    from trt_backend import TensorRTBackend

    events = []
    engine = FakeEngine(events, kind=kind, **engine_options)
    trt = FakeTrt(engine)
    cuda = FakeCuda(events)
    backend = TensorRTBackend(b"serialized-engine", kind, trt, cuda)
    return backend, engine, trt, cuda, events


class ImportAndCudaRuntimeTests(unittest.TestCase):
    def test_import_does_not_load_tensorrt_or_cudart(self):
        script = """
import builtins, ctypes, sys
real_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'tensorrt':
        raise AssertionError('TensorRT imported eagerly')
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
ctypes.CDLL = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('CUDA loaded eagerly'))
sys.path.insert(0, %r)
import trt_backend
print('safe')
""" % str(ROOT)
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=False
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "safe")

    def test_factory_reports_missing_tensorrt_only_when_called(self):
        import trt_backend

        with tempfile.NamedTemporaryFile() as engine_file:
            with mock.patch.object(
                trt_backend.importlib,
                "import_module",
                side_effect=ImportError("no tensorrt"),
            ):
                with self.assertRaisesRegex(trt_backend.BackendUnavailable, "TensorRT"):
                    trt_backend.create_backend(engine_file.name, "yolo")

    def test_factory_reports_missing_cuda_runtime_symbol(self):
        import trt_backend

        class IncompleteLibrary:
            cudaSetDevice = FakeFunction()

        with tempfile.NamedTemporaryFile() as engine_file:
            with mock.patch.object(
                trt_backend.importlib, "import_module", return_value=object()
            ), mock.patch.object(
                trt_backend.ctypes, "CDLL", return_value=IncompleteLibrary()
            ), mock.patch.object(trt_backend, "_cuda_runtime", None):
                with self.assertRaisesRegex(
                    trt_backend.BackendUnavailable, "CUDA Runtime"
                ):
                    trt_backend.create_backend(engine_file.name, "yolo")

    def test_cuda_runtime_declares_exact_signatures_and_preserves_large_pointers(self):
        from trt_backend import CudaRuntime

        library = FakeLibrary()
        cuda = CudaRuntime(library)
        pointer = cuda.malloc(64)
        stream = cuda.create_stream()
        cuda.memcpy_async(pointer, 2**43 + 1, 64, 1, stream)

        self.assertEqual(pointer, 2**40 + 123)
        self.assertEqual(stream, 2**41 + 7)
        self.assertEqual(library.cudaSetDevice.argtypes, [ctypes.c_int])
        self.assertEqual(library.cudaMalloc.argtypes, [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t])
        self.assertEqual(library.cudaFree.argtypes, [ctypes.c_void_p])
        self.assertEqual(
            library.cudaMemcpyAsync.argtypes,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p],
        )
        self.assertEqual(
            library.cudaStreamCreateWithFlags.argtypes,
            [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint],
        )
        self.assertEqual(library.cudaStreamSynchronize.argtypes, [ctypes.c_void_p])
        self.assertEqual(library.cudaStreamDestroy.argtypes, [ctypes.c_void_p])
        self.assertEqual(library.cudaGetErrorString.argtypes, [ctypes.c_int])
        for function in (
            library.cudaSetDevice,
            library.cudaMalloc,
            library.cudaFree,
            library.cudaMemcpyAsync,
            library.cudaStreamCreateWithFlags,
            library.cudaStreamSynchronize,
            library.cudaStreamDestroy,
        ):
            self.assertIs(function.restype, ctypes.c_int)
        self.assertIs(library.cudaGetErrorString.restype, ctypes.c_char_p)
        self.assertEqual(library.cudaStreamCreateWithFlags.calls[0][1].value, 1)
        memcpy_args = library.cudaMemcpyAsync.calls[0]
        self.assertEqual(memcpy_args[0].value, 2**40 + 123)
        self.assertEqual(memcpy_args[1].value, 2**43 + 1)

    def test_cuda_error_includes_operation_code_and_runtime_text(self):
        from trt_backend import CudaError, CudaRuntime

        cuda = CudaRuntime(FakeLibrary(error_code=17))
        with self.assertRaisesRegex(
            CudaError, r"cudaSetDevice failed \(17\): injected CUDA failure"
        ):
            cuda.set_device(0)


class BackendContractTests(unittest.TestCase):
    def test_backend_rejects_nonzero_cuda_device(self):
        from trt_backend import TensorRTBackend

        events = []
        engine = FakeEngine(events)
        with self.assertRaisesRegex(ValueError, "device 0"):
            TensorRTBackend(
                b"engine", "yolo", FakeTrt(engine), FakeCuda(events), device=1
            )
        self.assertEqual(events, [])

    def test_static_yolo_contract_uses_binding_dtypes_and_exact_nbytes(self):
        backend, engine, trt, cuda, events = make_backend(
            "yolo", input_dtype="fp16", output_dtype="fp32"
        )
        self.addCleanup(backend.close)

        self.assertEqual(backend.input_shape, (1, 3, 320, 320))
        self.assertEqual(backend.output_shape, (1, 27, 2100))
        self.assertEqual(backend.input_dtype, np.dtype(np.float16))
        self.assertEqual(backend.output_dtype, np.dtype(np.float32))
        self.assertEqual(backend.input_nbytes, 1 * 3 * 320 * 320 * 2)
        self.assertEqual(backend.output_nbytes, 1 * 27 * 2100 * 4)
        self.assertIs(backend.logger.severity, FakeLogger.ERROR)
        self.assertIs(backend.runtime, trt.runtime)
        self.assertIs(backend.engine, engine)
        self.assertIs(backend.context, engine.context)

    def test_legacy_engine_uses_640_input_and_26_class_output(self):
        import trt_backend

        events = []
        engine = FakeEngine(
            events, kind="yolo", input_shape=(1, 3, 640, 640), output_shape=(1, 30, 8400)
        )
        with tempfile.TemporaryDirectory() as directory:
            engine_path = Path(directory) / "test.engine"
            engine_path.write_bytes(b"engine")
            with mock.patch.object(trt_backend.importlib, "import_module", return_value=FakeTrt(engine)):
                with mock.patch.object(trt_backend, "_load_cuda_runtime", return_value=FakeCuda(events)):
                    backend = trt_backend.create_backend(engine_path, "yolo_legacy")
        self.addCleanup(backend.close)
        output = backend.infer(np.zeros((1, 3, 640, 640), dtype=np.float32))
        self.assertEqual(output.shape, (1, 30, 8400))
        with self.assertRaises(trt_backend.EngineContractError):
            trt_backend.TensorRTBackend(b"engine", "yolo", FakeTrt(engine), FakeCuda([]))

    def test_legacy_engine_rejects_bad_bindings_before_buffer_allocation(self):
        from trt_backend import EngineContractError, TensorRTBackend

        for options in (
            {"input_name": "input"},
            {"input_shape": (-1, 3, 640, 640)},
            {"output_shape": (1, 30, -1)},
            {"output_shape": (1, 8400, 30)},
        ):
            with self.subTest(options=options):
                events = []
                shapes = dict(input_shape=(1, 3, 640, 640), output_shape=(1, 30, 8400))
                shapes.update(options)
                engine = FakeEngine(events, kind="yolo", **shapes)
                with self.assertRaises(EngineContractError):
                    TensorRTBackend(b"engine", "yolo_legacy", FakeTrt(engine), FakeCuda(events))
                self.assertFalse(any(event[0] == "malloc" for event in events))

    def test_win8_engine_uses_fixed_640_input_and_8_class_output(self):
        from trt_backend import TensorRTBackend

        events = []
        input_shape, output_shape = (1, 3, 640, 640), (1, 12, 8400)
        engine = FakeEngine(events, input_shape=input_shape, output_shape=output_shape)
        backend = TensorRTBackend(b"engine", "yolo_win8", FakeTrt(engine), FakeCuda(events))
        self.addCleanup(backend.close)
        self.assertEqual(backend.infer(np.zeros(input_shape, dtype=np.float32)).shape, output_shape)

    def test_win8_engine_rejects_mismatched_input_or_output_contract_before_allocating(self):
        from trt_backend import EngineContractError, TensorRTBackend

        for kind, input_shape, output_shape in (
            ("yolo_win8", (1, 3, 320, 320), (1, 12, 8400)),
            ("yolo_win8", (1, 3, 640, 640), (1, 30, 8400)),
            ("yolo_legacy", (1, 3, 640, 640), (1, 12, 8400)),
        ):
            with self.subTest(kind=kind, input_shape=input_shape, output_shape=output_shape):
                events = []
                engine = FakeEngine(events, input_shape=input_shape, output_shape=output_shape)
                with self.assertRaisesRegex(EngineContractError, "unexpected YOLO (input|output) shape"):
                    TensorRTBackend(b"engine", kind, FakeTrt(engine), FakeCuda(events))
                self.assertFalse(any(event[0] == "malloc" for event in events))

    def test_yolo_requires_images_input_binding_name(self):
        from trt_backend import EngineContractError, TensorRTBackend

        events = []
        engine = FakeEngine(events, kind="yolo", input_name="input")
        with self.assertRaisesRegex(EngineContractError, "images"):
            TensorRTBackend(b"engine", "yolo", FakeTrt(engine), FakeCuda(events))

    def test_mask_requires_input_binding_name(self):
        from trt_backend import EngineContractError, TensorRTBackend

        events = []
        engine = FakeEngine(events, kind="mask", input_name="images")
        with self.assertRaisesRegex(EngineContractError, "input"):
            TensorRTBackend(b"engine", "mask", FakeTrt(engine), FakeCuda(events))

    def test_mask_requires_fixed_profile_zero_shape(self):
        from trt_backend import EngineContractError, TensorRTBackend

        events = []
        engine = FakeEngine(
            events,
            kind="mask",
            profile_shapes=(
                (1, 3, 160, 160),
                (1, 3, 320, 320),
                (1, 3, 640, 640),
            ),
        )
        with self.assertRaisesRegex(EngineContractError, "profile 0"):
            TensorRTBackend(b"engine", "mask", FakeTrt(engine), FakeCuda(events))

    def test_static_yolo_rejects_wrong_shape_before_allocating(self):
        from trt_backend import EngineContractError, TensorRTBackend

        events = []
        engine = FakeEngine(events, input_shape=(1, 3, 640, 640))
        with self.assertRaisesRegex(EngineContractError, "input.*shape"):
            TensorRTBackend(b"engine", "yolo", FakeTrt(engine), FakeCuda(events))
        self.assertNotIn("malloc", [event[0] for event in events])
        self.assertIn("destroy_stream", [event[0] for event in events])

    def test_rejects_implicit_batch_and_multiple_profiles(self):
        from trt_backend import EngineContractError, TensorRTBackend

        for options, message in [
            ({"implicit": True}, "implicit"),
            ({"profiles": 2}, "profile"),
        ]:
            with self.subTest(options=options):
                events = []
                engine = FakeEngine(events, **options)
                with self.assertRaisesRegex(EngineContractError, message):
                    TensorRTBackend(b"engine", "yolo", FakeTrt(engine), FakeCuda(events))

    def test_dynamic_mask_sets_profile_then_shape_before_resolving_output(self):
        backend, engine, trt, cuda, events = make_backend("mask")
        self.addCleanup(backend.close)
        events.clear()

        result = backend.infer(np.zeros((1, 3, 320, 320), dtype=np.float32))

        names = [event[0] for event in events]
        self.assertLess(names.index("profile"), names.index("set_shape"))
        self.assertLess(names.index("set_shape"), names.index("all_shapes"))
        self.assertLess(names.index("all_shapes"), names.index("output_shape"))
        self.assertLess(names.index("output_shape"), names.index("execute"))
        self.assertEqual(result.shape, (1, 1, 320, 320))

    def test_dynamic_mask_rejects_incomplete_or_unresolved_shapes(self):
        from trt_backend import EngineContractError, TensorRTBackend

        cases = [(False, (1, 1, 320, 320), "not all"), (True, (1, 1, -1, -1), "output")]
        for specified, output_shape, message in cases:
            with self.subTest(specified=specified, output_shape=output_shape):
                events = []
                context = FakeContext(events, output_shape)
                context.shapes_specified = specified
                engine = FakeEngine(events, kind="mask", context=context)
                with self.assertRaisesRegex(EngineContractError, message):
                    TensorRTBackend(b"engine", "mask", FakeTrt(engine), FakeCuda(events))

    def test_dynamic_mask_rejects_profile_or_binding_shape_setup_failure(self):
        from trt_backend import EngineContractError, TensorRTBackend

        cases = [
            ({"profile_result": False}, "profile 0"),
            ({"shape_result": False}, "outside profile 0"),
        ]
        for context_options, message in cases:
            with self.subTest(context_options=context_options):
                events = []
                context = FakeContext(
                    events, (1, 1, 320, 320), **context_options
                )
                engine = FakeEngine(events, kind="mask", context=context)
                with self.assertRaisesRegex(EngineContractError, message):
                    TensorRTBackend(b"engine", "mask", FakeTrt(engine), FakeCuda(events))


class BackendInferenceAndLifecycleTests(unittest.TestCase):
    def test_factory_provenance_identifies_loaded_bytes_without_reading_engine_again(self):
        import trt_backend

        for path_type in (str, Path):
            with self.subTest(path_type=path_type), tempfile.TemporaryDirectory() as directory:
                events = []
                engine_path = Path(directory) / 'loaded.engine'
                engine_path.write_bytes(b'abc')
                expected_path = str(engine_path.resolve())
                engine = FakeEngine(events)

                class ConsumingRuntime(FakeRuntime):
                    def deserialize_cuda_engine(self, engine_bytes):
                        value = super().deserialize_cuda_engine(engine_bytes)
                        engine_path.unlink()
                        return value

                class ConsumingTrt(FakeTrt):
                    def Runtime(self, logger):
                        self.runtime = ConsumingRuntime(self, logger)
                        return self.runtime

                trt = ConsumingTrt(engine)
                with mock.patch.object(trt_backend.importlib, 'import_module', return_value=trt), \
                        mock.patch.object(trt_backend, '_load_cuda_runtime', return_value=FakeCuda(events)):
                    backend = trt_backend.create_backend(path_type(engine_path), 'yolo')
                self.addCleanup(backend.close)
                self.assertEqual(trt.deserialized, b'abc')
                self.assertEqual(getattr(backend, 'provenance', None), {
                    'engine_path': expected_path,
                    'engine_sha256': 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad',
                    'model_kind': 'yolo',
                })
                backend.close()

    def test_infer_converts_input_and_orders_copy_execute_copy_sync(self):
        backend, engine, trt, cuda, events = make_backend("yolo", input_dtype="fp16")
        self.addCleanup(backend.close)
        source = np.ones((1, 3, 320, 640), dtype=np.float64)[..., ::2]
        self.assertFalse(source.flags.c_contiguous)
        events.clear()

        result = backend.infer(source)

        names = [event[0] for event in events]
        self.assertEqual(names, ["memcpy", "execute", "memcpy", "sync"])
        h2d, execute, d2h, sync = events
        self.assertEqual(h2d[3], backend.input_nbytes)
        self.assertEqual(h2d[4], 1)
        self.assertEqual(d2h[3], backend.output_nbytes)
        self.assertEqual(d2h[4], 2)
        self.assertEqual(execute[1][backend.input_index], backend.input_device_pointer)
        self.assertEqual(execute[1][backend.output_index], backend.output_device_pointer)
        self.assertGreater(execute[1][backend.input_index], 2**32)
        self.assertEqual(result.dtype, np.float32)
        self.assertFalse(np.shares_memory(result, backend._host_output))

    def test_execute_false_is_an_immediate_error(self):
        from trt_backend import TensorRTBackend, TensorRTExecutionError

        events = []
        context = FakeContext(events, (1, 27, 2100), execute_result=False)
        engine = FakeEngine(events, context=context)
        backend = TensorRTBackend(b"engine", "yolo", FakeTrt(engine), FakeCuda(events))
        self.addCleanup(backend.close)
        events.clear()

        with self.assertRaisesRegex(TensorRTExecutionError, "returned False"):
            backend.infer(np.zeros((1, 3, 320, 320), dtype=np.float16))
        self.assertEqual([event[0] for event in events], ["memcpy", "execute"])

    def test_input_shape_and_non_numeric_dtype_are_rejected_before_cuda(self):
        backend, engine, trt, cuda, events = make_backend("yolo")
        self.addCleanup(backend.close)
        for value, message in [
            (np.zeros((3, 320, 320), dtype=np.float16), "shape"),
            (np.full((1, 3, 320, 320), "x", dtype=object), "dtype"),
        ]:
            with self.subTest(message=message):
                events.clear()
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    backend.infer(value)
                self.assertEqual(events, [])

    def test_infer_and_close_are_rejected_from_non_owner_thread(self):
        backend, engine, trt, cuda, events = make_backend("yolo")
        failures = []

        def wrong_thread_calls():
            for action in (
                lambda: backend.infer(np.zeros((1, 3, 320, 320), dtype=np.float16)),
                backend.close,
            ):
                try:
                    action()
                except Exception as exc:
                    failures.append(exc)

        worker = threading.Thread(target=wrong_thread_calls)
        worker.start()
        worker.join()
        self.assertEqual(len(failures), 2)
        self.assertTrue(all("owner thread" in str(exc) for exc in failures))
        backend.close()

    def test_close_is_idempotent_and_releases_resources_in_reverse_order(self):
        backend, engine, trt, cuda, events = make_backend("yolo")
        input_pointer = backend.input_device_pointer
        output_pointer = backend.output_device_pointer
        stream = backend.stream
        events.clear()

        backend.close()
        backend.close()

        self.assertEqual(
            events,
            [
                ("sync", stream),
                ("free", output_pointer),
                ("free", input_pointer),
                ("destroy_stream", stream),
            ],
        )
        self.assertIsNone(backend.context)
        self.assertIsNone(backend.engine)
        self.assertIsNone(backend.runtime)
        self.assertIsNone(backend.logger)

    def test_partial_initialization_failure_cleans_allocations_and_stream(self):
        from trt_backend import CudaError, TensorRTBackend

        events = []
        engine = FakeEngine(events)

        class FailSecondMallocCuda(FakeCuda):
            def malloc(self, nbytes):
                if len(self.allocations) == 1:
                    self._record("malloc", nbytes)
                    raise CudaError("second allocation failed")
                return super().malloc(nbytes)

        cuda = FailSecondMallocCuda(events)
        with self.assertRaisesRegex(CudaError, "second allocation failed"):
            TensorRTBackend(b"engine", "yolo", FakeTrt(engine), cuda)

        names = [event[0] for event in events]
        self.assertIn(("free", cuda.allocations[0]), events)
        self.assertEqual(names.count("destroy_stream"), 1)

    def test_h2d_and_d2h_failures_can_be_closed_without_leaks(self):
        from trt_backend import CudaError, TensorRTBackend

        for failing_copy in (1, 2):
            with self.subTest(failing_copy=failing_copy):
                events = []
                engine = FakeEngine(events)

                class FailSelectedCopyCuda(FakeCuda):
                    def __init__(self, event_log):
                        super().__init__(event_log)
                        self.copy_count = 0

                    def memcpy_async(self, destination, source, nbytes, kind, stream):
                        self.copy_count += 1
                        self._record("memcpy", destination, source, nbytes, kind, stream)
                        if self.copy_count == failing_copy:
                            raise CudaError("copy %d failed" % failing_copy)

                cuda = FailSelectedCopyCuda(events)
                backend = TensorRTBackend(b"engine", "yolo", FakeTrt(engine), cuda)
                output_pointer = backend.output_device_pointer
                input_pointer = backend.input_device_pointer
                stream = backend.stream
                events.clear()
                with self.assertRaisesRegex(CudaError, "copy"):
                    backend.infer(
                        np.zeros((1, 3, 320, 320), dtype=np.float16)
                    )
                backend.close()
                self.assertIn(("free", output_pointer), events)
                self.assertIn(("free", input_pointer), events)
                self.assertIn(("destroy_stream", stream), events)

    def test_final_sync_failure_can_be_closed_and_released(self):
        from trt_backend import CudaError, TensorRTBackend

        events = []
        engine = FakeEngine(events)

        class FailFirstSyncCuda(FakeCuda):
            def __init__(self, event_log):
                super().__init__(event_log)
                self.fail_next_sync = False

            def synchronize(self, stream):
                self._record("sync", stream)
                if self.fail_next_sync:
                    self.fail_next_sync = False
                    raise CudaError("inference sync failed")

        cuda = FailFirstSyncCuda(events)
        backend = TensorRTBackend(b"engine", "yolo", FakeTrt(engine), cuda)
        output_pointer = backend.output_device_pointer
        input_pointer = backend.input_device_pointer
        stream = backend.stream
        cuda.fail_next_sync = True
        with self.assertRaisesRegex(CudaError, "inference sync failed"):
            backend.infer(np.zeros((1, 3, 320, 320), dtype=np.float16))
        backend.close()
        self.assertIn(("free", output_pointer), events)
        self.assertIn(("free", input_pointer), events)
        self.assertIn(("destroy_stream", stream), events)

    def test_close_continues_cleanup_and_raises_first_cleanup_error(self):
        from trt_backend import CudaError, TensorRTBackend

        events = []
        engine = FakeEngine(events)

        class FailCleanupCuda(FakeCuda):
            def synchronize(self, stream):
                self._record("sync", stream)
                raise CudaError("sync cleanup failed")

            def free(self, pointer):
                self._record("free", pointer)
                raise CudaError("free cleanup failed")

            def destroy_stream(self, stream):
                self._record("destroy_stream", stream)
                raise CudaError("destroy cleanup failed")

        cuda = FailCleanupCuda(events)
        backend = TensorRTBackend(b"engine", "yolo", FakeTrt(engine), cuda)
        output_pointer = backend.output_device_pointer
        input_pointer = backend.input_device_pointer
        stream = backend.stream
        events.clear()
        with self.assertRaisesRegex(CudaError, "sync cleanup failed"):
            backend.close()
        self.assertEqual(
            events,
            [
                ("sync", stream),
                ("free", output_pointer),
                ("free", input_pointer),
                ("destroy_stream", stream),
            ],
        )
        events.clear()
        backend.close()
        self.assertEqual(events, [])

    def test_initialization_error_is_not_masked_by_cleanup_errors(self):
        from trt_backend import CudaError, TensorRTBackend

        events = []
        engine = FakeEngine(events)

        class FailInitAndCleanupCuda(FakeCuda):
            def malloc(self, nbytes):
                if self.allocations:
                    self._record("malloc", nbytes)
                    raise CudaError("original allocation failure")
                return super().malloc(nbytes)

            def synchronize(self, stream):
                self._record("sync", stream)
                raise CudaError("cleanup sync failure")

            def free(self, pointer):
                self._record("free", pointer)
                raise CudaError("cleanup free failure")

            def destroy_stream(self, stream):
                self._record("destroy_stream", stream)
                raise CudaError("cleanup destroy failure")

        cuda = FailInitAndCleanupCuda(events)
        with self.assertRaisesRegex(CudaError, "original allocation failure"):
            TensorRTBackend(b"engine", "yolo", FakeTrt(engine), cuda)
        self.assertIn(("free", cuda.allocations[0]), events)
        self.assertIn("destroy_stream", [event[0] for event in events])


if __name__ == "__main__":
    unittest.main()

class TrayContractTests(unittest.TestCase):
    def test_tray_contract(self):
        from trt_backend import TensorRTBackend, EngineContractError
        events = []
        engine = FakeEngine(events, kind='yolo', input_shape=(1,3,640,640), output_shape=(1,5,8400))
        backend = TensorRTBackend(b'engine', 'tray', FakeTrt(engine), FakeCuda(events))
        self.addCleanup(backend.close)
        self.assertEqual(backend.infer(np.zeros((1,3,640,640), dtype=np.float32)).shape, (1,5,8400))
        with self.assertRaises(EngineContractError):
            TensorRTBackend(b'engine', 'yolo_legacy', FakeTrt(engine), FakeCuda([]))
