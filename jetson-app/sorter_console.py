"""Hardware-free Qt console for exercising the sorter workflow."""
import argparse
import json
import os
import tempfile
from pathlib import Path

from app import QApplication, ImageView, QDialog, QHBoxLayout, QLabel, QMainWindow, QPushButton, QTimer, QVBoxLayout, QWidget, Qt, STYLE, panel

try:
    from PySide6.QtCore import QLocale
    from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QFileDialog, QHeaderView, QStackedWidget, QStyledItemDelegate, QTableWidget, QTableWidgetItem, QTextEdit
except ImportError:
    from PyQt5.QtCore import QLocale
    from PyQt5.QtWidgets import QComboBox, QDoubleSpinBox, QFileDialog, QHeaderView, QStackedWidget, QStyledItemDelegate, QTableWidget, QTableWidgetItem, QTextEdit

import cv2
import numpy as np
import yaml

from camera_service import CameraError, CameraSource
from calibration_frame import CalibrationFrameView, frame_grid, validate_frame, normalize_frame_settings
from motion_limits import MACHINE_REGION, X_MAX_MM, Y_MAX_MM, position_in_machine
from media import MediaSource
from sorter_simulation import SimulatedSorter
from tray_locator import TrayLocator
from xy_calibration import CalibrationPoint, XYCalibration

ROOT = Path(__file__).resolve().parent
ROLES = ("left", "right", "top")
DEFAULT_POINTS = [
    (f"P{j * 3 + i + 1}", 211 + 133 * i, 92 + 133 * j, 24 + 56 * i, 24 + 56 * j)
    for j in range(3) for i in range(3)
]
TRAY_ROI = (154, 35, 379, 380)


def fit_calibration(rows, *, roi=TRAY_ROI, physical_region=MACHINE_REGION):
    if any(not position_in_machine(x, y) for _, _, _, x, y in rows):
        raise ValueError(f'机械坐标须在 X 0–{X_MAX_MM:g}、Y 0–{Y_MAX_MM:g} mm 范围内')
    left, top, width, height = roi
    return XYCalibration.fit(
        [CalibrationPoint(px, py, x, y) for _, px, py, x, y in rows],
        pixel_region=(left, top, left + width, top + height),
        physical_region=physical_region,
    )


class SimulatedDetection:
    """Fixture box in ROI pixels, matching the object drawn on the sample."""

    def detect(self, frame):
        return [(146, 160, 231, 225, .90)]


def button(label, callback):
    control = QPushButton(label)
    control.clicked.connect(callback)
    return control


def calibration_rows(table):
    rows = []
    for row in range(9):
        point_id = f"P{row + 1}"
        values = [float(table.item(row, col).text()) for col in range(1, 5)]
        if not all(np.isfinite(values)):
            raise ValueError(f"{point_id} 必须填写有限数字")
        px, py, x, y = values
        if not (0 <= px < 640 and 0 <= py < 480 and position_in_machine(x, y)):
            raise ValueError(f'{point_id} 超出画幅或机械行程：X 0–{X_MAX_MM:g}、Y 0–{Y_MAX_MM:g} mm')
        rows.append((point_id, *values))
    return rows


def legacy_rows(path):
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("version") not in (2, 3) or data.get("unit") != "mm":
        raise ValueError("请选择 v2 或 v3 毫米标定文件")
    try:
        points = data["points"]
        by_id = {point["id"]: point for point in points}
        ids = [f"P{i}" for i in range(1, 10)]
        if len(points) != 9 or set(by_id) != set(ids):
            raise ValueError("标定文件须包含 P1 至 P9 各一个")
        rows = []
        for name in ids:
            pixel, machine = by_id[name]['pixel'], by_id[name]['machine']
            if not isinstance(pixel, list) or not isinstance(machine, list):
                raise ValueError(f'{name} 须包含像素 X/Y 和机械 X/Y')
            px, py = map(float, pixel)
            x, y = map(float, machine)
            if not (0 <= px < 640 and 0 <= py < 480 and position_in_machine(x, y)):
                raise ValueError(f'{name} 超出画幅或机械行程：X 0–{X_MAX_MM:g}、Y 0–{Y_MAX_MM:g} mm')
            rows.append((name, px, py, x, y))
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError("旧版标定点字段不完整") from exc
    fit_calibration(rows)
    return rows


def sample_frame(role):
    frame = np.full((480, 640, 3), (35, 43, 51), dtype=np.uint8)
    if role == "top":
        cv2.rectangle(frame, (154, 35), (533, 415), (0, 195, 255), 2)
        cv2.rectangle(frame, (300, 195), (385, 260), (220, 150, 80), 2)
    else:
        cv2.circle(frame, (320, 235), 78, (190, 137, 43), 18)
    cv2.putText(frame, role.upper() + " / SIM", (18, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (236, 236, 236), 2)
    return frame


class CoordinateSpinBox(QDoubleSpinBox):
    def __init__(self, parent, table, row, column):
        super().__init__(parent)
        self.table, self.row, self.column = table, row, column
        self.setObjectName(f'coordinate_{row}_{column}')
        self.setLocale(QLocale.c())
        self.setDecimals(6)
        maximum = {1: 639.999999, 2: 479.999999, 3: X_MAX_MM, 4: Y_MAX_MM}[column]
        self.setRange(0, maximum)
        self.setSingleStep(1 if column < 3 else 0.1)
        self.setToolTip('每步 1 像素' if column < 3 else '每步 0.1 mm；点击 GOTO 才移动')

    def textFromValue(self, value):
        return format(value, '.6f').rstrip('0').rstrip('.')

    def focusInEvent(self, event):
        self.table.setCurrentCell(self.row, self.column)
        super().focusInEvent(event)

    def wheelEvent(self, event):
        self.setFocus()
        super().wheelEvent(event)


class CoordinateDelegate(QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        editor = CoordinateSpinBox(parent, self.parent(), index.row(), index.column())
        editor.valueChanged.connect(lambda: self.commitData.emit(editor))
        return editor

    def setEditorData(self, editor, index):
        try:
            value = float(index.data())
        except (TypeError, ValueError):
            return
        if not np.isfinite(value):
            return
        blocked = editor.blockSignals(True)
        try:
            editor.setValue(value)
        finally:
            editor.blockSignals(blocked)

    def setModelData(self, editor, model, index):
        role = getattr(Qt, 'EditRole', None)
        if role is None:
            role = Qt.ItemDataRole.EditRole
        model.setData(index, editor.value(), role)

    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(option.rect)


class CalibrationDialog(QDialog):
    def __init__(self, console, rows):
        super().__init__(console)
        self.console = console
        self.simulation = getattr(console, 'simulation_mode', True)
        settings = normalize_frame_settings(getattr(console, 'calibration_frame', None))
        self.setWindowTitle("XY 九点标定 · 仿真" if self.simulation else "XY 九点标定 · 实机")
        self.resize(1000, 550)
        outer = QVBoxLayout(self)
        frame_bar = QHBoxLayout()
        outer.addLayout(frame_bar)
        self.frame_fields = []
        for label, value in zip(('框 X', '框 Y', '框宽', '框高'), settings['box']):
            frame_bar.addWidget(QLabel(label))
            editor = QDoubleSpinBox()
            editor.setLocale(QLocale.c())
            editor.setDecimals(2)
            editor.setRange(0, 639)
            editor.setSingleStep(1)
            editor.setValue(value)
            frame_bar.addWidget(editor)
            self.frame_fields.append(editor)
        frame_bar.addWidget(QLabel('九点内缩'))
        self.frame_inset = QDoubleSpinBox()
        self.frame_inset.setRange(0, 45)
        self.frame_inset.setSuffix('%')
        self.frame_inset.setValue(settings['inset_percent'])
        frame_bar.addWidget(self.frame_inset)
        frame_bar.addWidget(button('按框生成九点', self.generate_points))
        work = QHBoxLayout()
        outer.addLayout(work, 1)
        self.view = CalibrationFrameView(settings['box'], self.set_frame)
        work.addWidget(panel('标定框 · 框内拖动平移 / 四角缩放 / 框外重画', self.view), 1)
        self.points = QTableWidget(9, 5)
        self.points.setHorizontalHeaderLabels(("点位", "像素 X", "像素 Y", "机械 X/mm", "机械 Y/mm"))
        self.points.setItemDelegate(CoordinateDelegate(self.points))
        header = self.points.horizontalHeader()
        stretch = getattr(QHeaderView, 'Stretch', None)
        if stretch is None:
            stretch = QHeaderView.ResizeMode.Stretch
        self.points.setColumnWidth(0, 40)
        for column in range(1, 5):
            header.setSectionResizeMode(column, stretch)
        self.set_rows(rows)
        self.points.currentCellChanged.connect(self.draw_points)
        self.points.itemChanged.connect(self.draw_points)
        work.addWidget(self.points, 1)
        self.status = QLabel(f'机械范围 X 0–{X_MAX_MM:g}、Y 0–{Y_MAX_MM:g} mm；修改后请点保存标定。'
                             '测试动作前需确认零点。')
        self.status.setWordWrap(True)
        outer.addWidget(self.status)
        controls = QHBoxLayout()
        controls.addWidget(button("GOTO 选中点", self.goto_selected))
        lift = QPushButton("按住下降对点")
        lift.pressed.connect(lambda: self.move_pusher(True))
        lift.released.connect(lambda: self.move_pusher(False))
        controls.addWidget(lift)
        controls.addWidget(button("抬起", lambda: self.move_pusher(False)))
        controls.addWidget(button("导入旧标定", self.open_legacy))
        controls.addWidget(button("保存标定", self.save_points))
        controls.addWidget(button("关闭", self.reject))
        outer.addLayout(controls)
        self.points.setCurrentCell(0, 1)
        self.set_frame(settings['box'])
        for editor in self.frame_fields:
            editor.valueChanged.connect(self.frame_fields_changed)
        # Keep the live top camera visible while adjusting and testing alignment.
        self.preview_timer = QTimer(self)
        self.preview_timer.timeout.connect(self.draw_points)
        self.preview_timer.start(150)
        self.draw_points()

    def set_frame(self, box):
        box = validate_frame(box)
        self.view.box = box
        x, y, width, height = box
        limits = ((0, 639 - width), (0, 479 - height), (2, 639 - x), (2, 479 - y))
        for editor, value, (minimum, maximum) in zip(self.frame_fields, box, limits):
            blocked = editor.blockSignals(True)
            try:
                editor.setRange(0, 639)
                editor.setValue(value)
                editor.setRange(minimum, maximum)
            finally:
                editor.blockSignals(blocked)
        self.draw_points()

    def frame_fields_changed(self, *_):
        self.set_frame(tuple(editor.value() for editor in self.frame_fields))

    def generate_points(self):
        try:
            pixels = frame_grid(self.view.box, self.frame_inset.value())
            calibration_rows(self.points)
            blocked = self.points.blockSignals(True)
            try:
                for row, (px, py) in enumerate(pixels):
                    for column, value in ((1, px), (2, py)):
                        self.points.item(row, column).setText(format(value, '.6f').rstrip('0').rstrip('.'))
            finally:
                self.points.blockSignals(blocked)
            self.draw_points()
            self.status.setText('已按框生成像素点草稿，机械 X/Y 保留；请逐点微调后保存。')
        except (AttributeError, ValueError) as exc:
            self.status.setText(str(exc))

    def set_rows(self, rows):
        blocked = self.points.blockSignals(True)
        try:
            for r, row in enumerate(rows):
                for c, value in enumerate(row):
                    previous = self.points.item(r, c)
                    if previous is not None and c > 0:
                        self.points.closePersistentEditor(previous)
                    item = QTableWidgetItem(str(value) if c == 0 else format(float(value), '.6f').rstrip('0').rstrip('.'))
                    if c == 0:
                        editable = getattr(Qt, 'ItemIsEditable', None)
                        if editable is None:
                            editable = Qt.ItemFlag.ItemIsEditable
                        item.setFlags(item.flags() & ~editable)
                    self.points.setItem(r, c, item)
            for row in range(9):
                for column in range(1, 5):
                    self.points.openPersistentEditor(self.points.item(row, column))
        finally:
            self.points.blockSignals(blocked)

    def open_legacy(self):
        folder = ROOT / 'imports'
        path, _ = QFileDialog.getOpenFileName(self, "导入旧版九点标定", str(folder) if folder.is_dir() else "", "标定文件 (*.yaml *.yml)")
        if path:
            self.import_points(path)

    def import_points(self, path):
        try:
            rows = legacy_rows(path)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            self.status.setText(f"导入失败：{exc}")
            return
        self.set_rows(rows)
        self.draw_points()
        self.status.setText("已导入草稿；请检查后保存。")

    def draw_points(self, *_):
        frame = cv2.resize(self.console.frames["top"], (640, 480))
        x, y, width, height = self.view.box
        left, top, right, bottom = map(round, (x, y, x + width, y + height))
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 220, 70), 2)
        for corner in ((left, top), (right, top), (left, bottom), (right, bottom)):
            cv2.rectangle(frame, (corner[0] - 4, corner[1] - 4),
                          (corner[0] + 4, corner[1] + 4), (255, 220, 70), -1)
        try:
            rows = calibration_rows(self.points)
        except (AttributeError, ValueError):
            self.view.show_frame(frame)
            return
        for i, (name, px, py, _, _) in enumerate(rows):
            x, y = round(px), round(py)
            color = (80, 245, 110) if i == self.points.currentRow() else (0, 185, 255)
            cv2.circle(frame, (x, y), 4, color, -1)
            cv2.putText(frame, name, (x + 6, y - 5), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
        self.view.show_frame(frame)

    def goto_selected(self):
        try:
            row = self.points.currentRow()
            if row < 0:
                raise ValueError("请选择一个标定点")
            name, _, _, x, y = calibration_rows(self.points)[row]
            self.console._command(lambda: self.console.machine.goto(x, y), f"GOTO {name} → {x:.1f}, {y:.1f} mm")
            self.status.setText(self.console.statusBar().currentMessage())
        except (AttributeError, ValueError, RuntimeError) as exc:
            self.status.setText(str(exc))

    def move_pusher(self, down):
        try:
            description = "推板下降" if down else "推板抬起"
            self.console._command(lambda: self.console.machine.set_pusher(down), description)
            self.status.setText(self.console.statusBar().currentMessage())
        except (ValueError, RuntimeError) as exc:
            self.status.setText(str(exc))

    def save_points(self):
        try:
            frame = {'box': list(self.view.box), 'inset_percent': self.frame_inset.value()}
            self.console.save_calibration(calibration_rows(self.points), frame=frame)
            self.status.setText('已保存九点、标定框和内缩比例；请实测九点后确认。')
        except (AttributeError, OSError, ValueError) as exc:
            self.status.setText(str(exc))

    def done(self, result):
        self.preview_timer.stop()
        if self.console.machine.snapshot().pusher_down or getattr(self.console.machine, 'pending', None) == 'down':
            self.move_pusher(False)
        super().done(result)


class SimulationWindow(QMainWindow):
    def __init__(self, calibration_path=None):
        super().__init__()
        self.machine = SimulatedSorter()
        self.calibration_path = Path(calibration_path) if calibration_path else ROOT / "data/simulation_calibration.json"
        self.calibration_frame = normalize_frame_settings()
        self.frames = {role: sample_frame(role) for role in ROLES}
        self.setWindowTitle("YLERS · Xavier NX 分拣仿真台")
        self.setStyleSheet(STYLE)
        self.resize(1024, 600)
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        banner = QLabel("仿真模式 · 机械输出关闭")
        banner.setStyleSheet("font-weight:bold;color:#0d6963")
        outer.addWidget(banner)
        self.pages = QStackedWidget()
        outer.addWidget(self.pages, 1)
        self.debug_page = QWidget()
        self.competition_page = QWidget()
        self.pages.addWidget(self.debug_page)
        self.pages.addWidget(self.competition_page)
        self._make_debug_page()
        self._make_competition_page()
        self.calibration_rows = self.load_calibration()
        self.write_log("已启动。先确认仿真零点。")

    def _make_debug_page(self):
        layout = QVBoxLayout(self.debug_page)
        cameras = QHBoxLayout()
        self.views = {}
        for role, label in zip(ROLES, ("左轴测", "右轴测", "顶部托盘")):
            view = self.views[role] = ImageView(label)
            view.show_frame(self.frames[role])
            group = QVBoxLayout()
            group.addWidget(panel(label, view), 1)
            group.addWidget(button("载入样本", lambda _=False, r=role: self.open_frame(r)))
            cameras.addLayout(group, 1)
        layout.addLayout(cameras, 1)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.document().setMaximumBlockCount(150)
        self.log_view.setFixedHeight(105)
        layout.addWidget(self.log_view)
        self.machine_status = QLabel()
        layout.addWidget(self.machine_status)
        controls = QHBoxLayout()
        controls.addWidget(button("确认仿真零点", self.confirm_origin))
        controls.addWidget(button("开始", self.start))
        controls.addWidget(button("停止", self.stop))
        controls.addWidget(button("九点标定", self.show_calibration))
        controls.addWidget(button("比赛页", self.show_competition))
        layout.addLayout(controls)

    def _make_competition_page(self):
        layout = QVBoxLayout(self.competition_page)
        self.total_label = QLabel("成功总数 0")
        self.total_label.setStyleSheet("font-size:24px;font-weight:bold;color:#0d6963")
        layout.addWidget(self.total_label)
        self.records = QTableWidget(0, 4)
        self.records.setHorizontalHeaderLabels(("序号", "货物", "储物盒", "累计成功"))
        self.records.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.records, 1)
        counts = QHBoxLayout()
        self.bin_labels = []
        for i in range(1, 7):
            label = QLabel(f"{i}号盒 · 0")
            self.bin_labels.append(label)
            counts.addWidget(label)
        layout.addLayout(counts)
        controls = QHBoxLayout()
        self.bin_picker = QComboBox()
        self.bin_picker.addItems([str(i) for i in range(1, 7)])
        controls.addWidget(QLabel("模拟分配箱号"))
        controls.addWidget(self.bin_picker)
        controls.addWidget(button("模拟分拣一件", self.run_cycle))
        controls.addWidget(button("调试页", self.show_debug))
        layout.addLayout(controls)

    def write_log(self, message):
        self.log_view.append(str(message))
        self.statusBar().showMessage(str(message))
        state = self.machine.snapshot()
        self.machine_status.setText(f"状态 {state.state} · XY {state.x_mm:.1f}, {state.y_mm:.1f} mm"
                                    f" · 推板{'下降' if state.pusher_down else '抬起'}")

    def open_frame(self, role):
        path, _ = QFileDialog.getOpenFileName(self, "打开图片或录像", "", "媒体 (*.png *.jpg *.jpeg *.bmp *.mp4 *.avi)")
        if not path:
            return
        source = (CameraSource(path) if Path(path).suffix.lower() in {".mp4", ".avi"}
                  else MediaSource())
        try:
            source.open() if isinstance(source, CameraSource) else source.open(path)
            frame = source.read()
            if frame is None:
                raise ValueError("没有可读取的画面")
            self.frames[role] = frame
            self.views[role].show_frame(frame)
            self.write_log(f"{role} 已载入样本：{Path(path).name}")
        except (CameraError, OSError, ValueError, cv2.error) as exc:
            self.write_log(str(exc))
        finally:
            try:
                source.close()
            except CameraError as exc:
                self.write_log(str(exc))

    def _command(self, command, description):
        try:
            if not command():
                self.write_log(self.machine.snapshot().last_error or "仿真操作未执行")
                return False
        except (ValueError, RuntimeError) as exc:
            self.write_log(str(exc))
            return False
        self.write_log(description)
        return True

    def confirm_origin(self):
        return self._command(self.machine.confirm_origin, "仿真零点已确认")

    def start(self):
        return self._command(self.machine.start, "开始仿真分拣")

    def stop(self):
        return self._command(self.machine.stop, "仿真分拣已停止")

    def run_cycle(self):
        bin_id = int(self.bin_picker.currentText())
        try:
            locator = TrayLocator(fit_calibration(self.calibration_rows), SimulatedDetection(), roi=TRAY_ROI)
            target = locator.locate(sample_frame("top"))
        except (ValueError, RuntimeError) as exc:
            self.write_log(f"仿真定位失败：{exc}")
            return
        if target is None:
            self.write_log("仿真托盘没有可用坐标")
            return
        if not self._command(lambda: self.machine.run_cycle(*target, bin_id), "收到仿真 DONE"):
            return
        snapshot = self.machine.snapshot()
        total = snapshot.completed_count
        self.total_label.setText(f"成功总数 {total}")
        self.bin_labels[bin_id - 1].setText(f"{bin_id}号盒 · {snapshot.bin_counts[bin_id - 1]}")
        for line in snapshot.log[-5:]:
            self.log_view.append(str(line))
        row = self.records.rowCount()
        self.records.insertRow(row)
        for col, value in enumerate((total, "仿真样件", bin_id, total)):
            self.records.setItem(row, col, QTableWidgetItem(str(value)))

    def show_competition(self):
        self.pages.setCurrentWidget(self.competition_page)

    def show_debug(self):
        self.pages.setCurrentWidget(self.debug_page)

    def load_calibration(self):
        rows = DEFAULT_POINTS
        try:
            if self.calibration_path.is_file():
                data = json.loads(self.calibration_path.read_text(encoding="utf-8"))
                if data["mode"] != "simulation":
                    raise ValueError("文件不是仿真标定")
                rows = data["points"]
                if len(rows) != 9 or any(len(row) != 5 for row in rows):
                    raise ValueError("需要九个标定点")
                fit_calibration(rows)
                self.calibration_frame = normalize_frame_settings(data.get('frame'))
        except (KeyError, OSError, TypeError, ValueError) as exc:
            rows = DEFAULT_POINTS
            self.write_log(f"仿真标定未加载：{exc}")
        return rows

    def create_calibration_dialog(self):
        return CalibrationDialog(self, self.calibration_rows)

    def show_calibration(self):
        dialog = self.create_calibration_dialog()
        try:
            dialog.exec()
        finally:
            dialog.deleteLater()

    def save_calibration(self, rows, frame=None):
        fit_calibration(rows)
        frame = normalize_frame_settings(self.calibration_frame if frame is None else frame)
        self.calibration_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, dir=str(self.calibration_path.parent)) as stream:
            temporary = Path(stream.name)
            try:
                json.dump({"mode": "simulation", "points": rows, 'frame': frame}, stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            except Exception:
                stream.close()
                temporary.unlink()
                raise
        try:
            os.replace(str(temporary), str(self.calibration_path))
            self.calibration_rows = rows
            self.calibration_frame = frame
        finally:
            if temporary.exists():
                temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description="Xavier NX 分拣仿真台")
    parser.add_argument("--fullscreen", action="store_true")
    args = parser.parse_args()
    app = QApplication([])
    window = SimulationWindow()
    window.showFullScreen() if args.fullscreen else window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
