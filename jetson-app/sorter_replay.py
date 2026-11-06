"""Run recorded camera frames through the sorter without opening a serial port."""

import argparse
import ast
import json
from pathlib import Path

import yaml

from camera_service import CameraService
from legacy_vision import LegacyRecognizer
from sorter_simulation import SimulatedSorter
from motion_limits import MACHINE_REGION, position_in_machine
from tray_locator import TrayDetector, TrayLocator
from vision import load_class_names
from xy_calibration import CalibrationPoint, XYCalibration


ROOT = Path(__file__).resolve().parent
ROLES = ("left", "right", "top")


class RecordedSorter:
    def __init__(self, machine, read_frame, locate, recognize, bindings):
        self.machine = machine
        self.read_frame = read_frame
        self.locate = locate
        self.recognize = recognize
        self.bindings = bindings

    def step(self):
        if self.machine.snapshot().state != "running":
            raise ValueError("simulated sorter must be running")
        try:
            frames = {role: self.read_frame(role) for role in ROLES}
            missing = [role for role, frame in frames.items() if frame is None]
            if missing:
                raise ValueError("no frame for " + ", ".join(missing))
            target = self.locate(frames["top"])
            if target is None:
                self.machine.handle_mcu("PUSH?")
                return "no_target"
            if not self.machine.begin_cycle(*target):
                raise ValueError(self.machine.snapshot().last_error or "PUSH rejected")
            result = self.recognize(frames["left"], frames["right"])
            if result.error:
                raise ValueError(result.error)
            if result.detection is None or result.color is None or not result.color.valid_pixels:
                raise ValueError("side recognition returned no usable shape/color")
            key = result.detection.class_name, result.color.name
            if not self.machine.complete_scan(self.bindings.get(key)):
                raise ValueError("no valid six-bin binding for {!r}".format(key))
            if not self.machine.handle_mcu("DONE"):
                raise ValueError("simulated DONE rejected")
            return "done"
        except Exception:
            self.machine.stop()
            raise


def _existing_file(value):
    path = Path(value)
    if not path.is_file():
        raise argparse.ArgumentTypeError("local file does not exist: " + value)
    return path


def _bindings(path, class_names):
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError("bindings must be a list")
    result = {}
    for row in data:
        if not isinstance(row, dict):
            raise ValueError("invalid binding")
        shape, color, bin_id = row.get("shape"), row.get("color"), row.get("bin")
        if (shape not in class_names or not isinstance(color, str) or not color.strip()
                or color == "未知颜色" or type(bin_id) is not int or not 1 <= bin_id <= 6):
            raise ValueError("invalid binding: {!r}".format(row))
        if (shape, color) in result:
            raise ValueError("duplicate binding: {!r}".format((shape, color)))
        result[(shape, color)] = bin_id
    return result


def _calibration(path):
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if data.get("mode") != "simulation":
        raise ValueError("expected a calibration JSON saved by the simulation console")
    rows = data["points"]
    if len(rows) != 9 or {row[0] for row in rows} != {"P{}".format(i) for i in range(1, 10)}:
        raise ValueError("calibration requires P1–P9")
    if any(not position_in_machine(row[3], row[4]) for row in rows):
        raise ValueError('calibration point outside machine travel')
    return XYCalibration.fit(
        [CalibrationPoint(*row[1:]) for row in rows],
        pixel_region=(154, 35, 533, 415),
        physical_region=MACHINE_REGION,
    )


class _OnnxCPU:
    def __init__(self, path, input_shape, output_shape):
        import onnxruntime

        self.session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if (len(inputs) != 1 or len(outputs) != 1 or inputs[0].name != "images"
                or inputs[0].shape != input_shape or outputs[0].shape != output_shape
                or inputs[0].type != "tensor(float)"):
            raise ValueError("unexpected ONNX contract: " + path.name)

    def infer(self, tensor):
        return self.session.run(None, {"images": tensor})[0]

    def close(self):
        self.session = None


def main():
    parser = argparse.ArgumentParser(description="Xavier NX recorded sorter rehearsal (no hardware)")
    for key in ROLES + ("shape-onnx", "tray-onnx", "calibration", "bindings"):
        parser.add_argument("--" + key, type=_existing_file, required=True)
    parser.add_argument("--steps", type=int, default=1, help="number of synchronized frame triplets to process")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if any(getattr(args, role).suffix.lower() not in {".avi", ".mp4", ".mkv", ".mov"} for role in ROLES):
        parser.error("all three camera inputs must be local video files")

    cameras = CameraService({role: getattr(args, role).resolve() for role in ROLES})
    side = tray = recognizer = None
    try:
        names = load_class_names(ROOT / "models/legacy_classes.txt", expected_count=26)
        bindings = _bindings(args.bindings, names)
        calibration = _calibration(args.calibration)
        side = _OnnxCPU(args.shape_onnx, [1, 3, 640, 640], [1, 30, 8400])
        model_names = ast.literal_eval(side.session.get_modelmeta().custom_metadata_map["names"])
        if model_names != dict(enumerate(names)):
            raise ValueError("shape model class order does not match legacy_classes.txt")
        tray = _OnnxCPU(args.tray_onnx, [1, 3, 640, 640], [1, 5, 8400])
        recognizer = LegacyRecognizer(
            side, None, names, yaml.safe_load((ROOT / "legacy_vision.yaml").read_text(encoding="utf-8"))
        )
        locator = TrayLocator(calibration, TrayDetector(tray), roi=(154, 35, 379, 380))

        def locate(frame):
            if frame.shape[:2] != (480, 640):
                raise ValueError("top video must be 640×480 to match the simulation calibration")
            return locator.locate(frame)

        for role in ROLES:
            cameras.open(role)
        machine = SimulatedSorter()
        machine.confirm_origin()
        machine.start()
        replay = RecordedSorter(machine, cameras.read, locate, recognizer.recognize_pair, bindings)
        for index in range(args.steps):
            status = replay.step()
            snapshot = machine.snapshot()
            print(json.dumps({"mode": "simulation", "frame": index + 1, "status": status,
                              "total": snapshot.completed_count, "bins": snapshot.bin_counts}))
    except Exception as exc:
        parser.exit(1, "Recorded rehearsal stopped: {}\n".format(exc))
    finally:
        cameras.close()
        if recognizer is not None:
            recognizer.close()
        elif side is not None:
            side.close()
        if tray is not None:
            tray.close()


if __name__ == "__main__":
    main()
