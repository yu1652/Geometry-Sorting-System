"""Physical Jetson console. Motion is acknowledged only by the ESP32."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from app import (QApplication, ImageView, QDialog, QHBoxLayout, QLabel,
                 QMainWindow, QPushButton, QVBoxLayout, QWidget, QTimer,
                 QMessageBox, QComboBox, QFormLayout, QDialogButtonBox,
                 QBUTTON_OK, QBUTTON_CANCEL, QDIALOG_ACCEPTED, STYLE, panel)
try:
    from PySide6.QtWidgets import QAbstractItemView, QCheckBox, QHeaderView, QStackedWidget, QTableWidget, QTableWidgetItem, QTextEdit
except ImportError:
    from PyQt5.QtWidgets import QAbstractItemView, QCheckBox, QHeaderView, QStackedWidget, QTableWidget, QTableWidgetItem, QTextEdit

import cv2
import numpy as np
import serial
from serial.tools import list_ports

from esp32_serial import SerialLineTransport, SerialTransportError
from live_control import LiveController
from live_vision import LiveVisionWorker
from legacy_vision import surface_summary
from calibration_frame import normalize_frame_settings
from motion_limits import X_MAX_MM, Y_MAX_MM, position_in_machine
from auto_bins import AutoBinSession, CapacityFull
from runtime_diagnostics import RuntimeDiagnostics
from scan_gate import ScanConsensus
from sorter_console import (CalibrationDialog, DEFAULT_POINTS, TRAY_ROI,
                            button, fit_calibration)

ROOT = Path(__file__).resolve().parent
ROLES = ('left', 'right', 'top')
CAMERA_TITLES = ('识别区俯拍 / 拼图左', '识别区侧视 / 拼图右', '托盘定位俯拍')
COLORS = ('红色', '橙色', '黄色', '绿色', '青色', '蓝色', '紫色', '黑色', '白色', '灰色')
AUTO_STATES = {'enabling_auto', 'starting', 'running', 'locating', 'wait_scan', 'recognizing', 'wait_done'}
DEFAULT_SOURCES = {
    role: '/dev/v4l/by-path/platform-3610000.xhci-usb-0:%s:1.0-video-index0' % port
    for role, port in zip(ROLES, ('2.1', '2.2', '2.4'))
}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf8', dir=str(path.parent), delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def validate_points(rows):
    if (not isinstance(rows, (list, tuple)) or len(rows) != 9
            or any(len(row) != 5 for row in rows)
            or {row[0] for row in rows} != {'P%s' % n for n in range(1, 10)}):
        raise ValueError('需要 P1 至 P9 各一个完整标定点')
    for _, px, py, x, y in rows:
        if not (0 <= px < 640 and 0 <= py < 480 and position_in_machine(x, y)):
            raise ValueError(f'标定点超出画幅或机械行程：X 0–{X_MAX_MM:g}、Y 0–{Y_MAX_MM:g} mm')
    return fit_calibration(rows)


class LiveWindow(QMainWindow):
    simulation_mode = False

    def __init__(self, data_dir=None, vision_factory=LiveVisionWorker, serial_factory=serial.Serial):
        super().__init__()
        self.data_dir = Path(data_dir) if data_dir else ROOT / 'data'
        self.calibration_path = self.data_dir / 'live_calibration.json'
        self.auto_bins_path = self.data_dir / 'live_auto_bins.json'
        self.device_path = self.data_dir / 'live_devices.json'
        self.diagnostics = RuntimeDiagnostics(self.data_dir)
        self._diagnostic_log_error = False
        self._scan_gate = None
        self._scan_epoch = None
        self._scan_next_at = 0
        self.calibration_frame = normalize_frame_settings()
        self.names = (ROOT / 'models/legacy_classes.txt').read_text(encoding='utf8').splitlines()
        self.sources = dict(DEFAULT_SOURCES)
        self._vision_factory, self._serial_factory = vision_factory, serial_factory
        self.machine = LiveController(self._send)
        self.serial_port = self.transport = self.vision = None
        self._handshake_at = None
        self._flushed = False
        self._disconnect_deadline = None
        self._reset_deadline = None
        self._heartbeat_at = time.monotonic()
        self._closing = False
        self._close_ready = False
        self._sequence = 0
        self._job = None
        self._last_sort = None
        self._model_pending = False
        self._model_process = None
        self._frames_at = None
        self.frames = {role: np.zeros((480, 640, 3), np.uint8) for role in ROLES}
        self.auto_bins = None
        self._binding_error = None
        self.calibration_valid = False
        self.calibration_rows = list(DEFAULT_POINTS)
        self.setWindowTitle('YLERS · Xavier NX 实机分拣操作台')
        self.setStyleSheet(STYLE)
        self.resize(1024, 680)
        self._make_ui()
        self._load_settings()
        self._load_auto_bins()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll_serial)
        self.timer.start(40)
        self.write_log('实机入口已打开。先连接相机与串口，手动对零后确认。')
        self.write_log('自动识别：模型就绪后重新取图，等待至少 0.8 秒；最多复核 8 次，连续两次形状和颜色一致才投箱。')

    def _make_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        self.banner = QLabel('实机模式 · 未确认零点 · 自动启动关闭')
        self.banner.setStyleSheet('font-weight:bold;color:#8a402c')
        outer.addWidget(self.banner)
        connection = QHBoxLayout()
        self.port_picker = QComboBox()
        self.port_picker.setEditable(True)
        self.refresh_ports()
        connection.addWidget(self.port_picker, 1)
        connection.addWidget(button('刷新串口', self.refresh_ports))
        self.serial_button = button('连接串口', self.connect_serial)
        connection.addWidget(self.serial_button)
        self.camera_button = button('连接三路相机', self.connect_cameras)
        connection.addWidget(self.camera_button)
        connection.addWidget(button('相机设置', self.camera_settings))
        connection.addWidget(button('模型管理', self.open_models))
        self.diagnostic_button = button('导出诊断', self.export_diagnostics)
        connection.addWidget(self.diagnostic_button)
        outer.addLayout(connection)
        self.pages = QStackedWidget()
        outer.addWidget(self.pages, 1)
        self.debug_page, self.competition_page = QWidget(), QWidget()
        self.pages.addWidget(self.debug_page)
        self.pages.addWidget(self.competition_page)
        self._make_debug_page()
        self._make_competition_page()
        controls = QHBoxLayout()
        self.actions = {}
        for label, callback in (
            ('手动零点确认', self.confirm_origin), ('九点标定 / 升降', self.show_calibration),
            ('本轮箱号', self.binding_dialog), ('清空本轮', self.clear_round),
            ('运行', self.start), ('停止', self.stop),
            ('复位', self.reset), ('调试页', self.show_debug), ('比赛页', self.show_competition),
            ('退出', self.close),
        ):
            control = self.actions[label] = button(label, callback)
            controls.addWidget(control)
        outer.addLayout(controls)
        self.machine_status = QLabel('串口未连接')
        outer.addWidget(self.machine_status)

    def _make_debug_page(self):
        layout = QVBoxLayout(self.debug_page)
        cameras = QHBoxLayout()
        self.views = {}
        for role, title in zip(ROLES, CAMERA_TITLES):
            view = self.views[role] = ImageView('相机未连接')
            cameras.addWidget(panel(title, view), 1)
        layout.addLayout(cameras, 2)
        middle = QHBoxLayout()
        self.preview = ImageView('识别拼接图')
        middle.addWidget(panel('拼接检测结果', self.preview), 1)
        info = QVBoxLayout()
        self.result_label = QLabel('等待识别')
        self.result_label.setWordWrap(True)
        info.addWidget(self.result_label)
        info.addWidget(button('手动识别（只显示）', self.manual_recognize))
        info.addWidget(button('托盘定位（只显示）', self.manual_locate))
        self.calibration_confirmed = QCheckBox('九点标定已在实机逐点核对')
        info.addWidget(self.calibration_confirmed)
        middle.addLayout(info, 1)
        layout.addLayout(middle, 1)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.document().setMaximumBlockCount(200)
        self.log_view.setFixedHeight(90)
        layout.addWidget(self.log_view)

    def _make_competition_page(self):
        layout = QVBoxLayout(self.competition_page)
        self.total_label = QLabel('本轮成功总数 0')
        self.total_label.setStyleSheet('font-size:24px;font-weight:bold;color:#147d73')
        layout.addWidget(self.total_label)
        self.records = QTableWidget(0, 4)
        self.records.setHorizontalHeaderLabels(('序号', '货物名称', '图片', '本轮成功总数'))
        header = self.records.horizontalHeader()
        stretch = getattr(QHeaderView, 'Stretch', None)
        if stretch is None:
            stretch = QHeaderView.ResizeMode.Stretch
        header.setSectionResizeMode(1, stretch)
        header.setSectionResizeMode(2, stretch)
        self.records.setColumnWidth(0, 60)
        self.records.setColumnWidth(3, 145)
        layout.addWidget(self.records, 1)
        boxes = QHBoxLayout()
        self.bin_labels = []
        for number in range(1, 7):
            label = QLabel('%s号盒 · 0' % number)
            label.setWordWrap(True)
            self.bin_labels.append(label)
            boxes.addWidget(label)
        layout.addLayout(boxes)

    def write_log(self, message):
        line = datetime.now().strftime('%H:%M:%S') + ' ' + str(message)
        self.log_view.append(line)
        self._persist_log(message)
        self.statusBar().showMessage(str(message))
        self.update_state()

    def _persist_log(self, message):
        try:
            self.diagnostics.log(message)
            self._diagnostic_log_error = False
        except (OSError, ValueError) as error:
            if not self._diagnostic_log_error:
                self.log_view.append('诊断日志保存失败：' + str(error))
                self._diagnostic_log_error = True

    def _save_diagnostic(self, result, token, kind):
        evidence = getattr(result, 'diagnostics', None) or {}
        context = {'token': token, 'epoch': self.machine.epoch, 'kind': kind,
                   'machine_state': self.machine.snapshot().state,
                   'scan_attempt': self._scan_gate.attempts + 1 if kind == 'sort' and self._scan_gate else None,
                   'preprocessed': evidence.get('preprocessed', False),
                   'class_names': evidence.get('class_names', self.names),
                   'recognizer_config': evidence.get('config'),
                   'models': evidence.get('models'),
                   'capture': evidence.get('capture'), 'sources': self.sources}
        try:
            path = self.diagnostics.save_sample(result, context)
            self.write_log('识别样本已保存：' + str(path))
        except (OSError, ValueError, TypeError, cv2.error) as error:
            self.write_log('识别样本保存失败：' + str(error))

    def export_diagnostics(self):
        if self.machine.snapshot().state in AUTO_STATES or self.machine.pending:
            self.write_log('请先停止自动流程并等待动作确认，再导出诊断。')
            return None
        try:
            path = self.diagnostics.export(
                extra_files=(ROOT / 'legacy_vision.yaml', ROOT / 'models/legacy_classes.txt',
                             self.calibration_path, self.device_path, self.auto_bins_path,
                             ROOT / 'data/model_store/active.json'),
                metadata={'application': 'YLERS live console', 'sources': self.sources,
                          'state': self.machine.snapshot().state, 'class_names': self.names})
            self.write_log('诊断包已导出：' + str(path))
            QMessageBox.information(self, '诊断已导出', '请把这个文件传回电脑：\n' + str(path))
            return path
        except (OSError, ValueError, TypeError) as error:
            self.write_log('诊断导出失败：' + str(error))
            return None

    def update_state(self):
        state = self.machine.snapshot()
        self.machine_status.setText('状态 %s · XY %.1f, %.1f mm · 升降 %s · %s' % (
            state.state, state.x_mm, state.y_mm, '下降' if state.pusher_down else '抬起',
            '动作等待确认' if self.machine.pending else ''))
        self.banner.setText('实机模式 · %s · %s · 自动分箱 %s/6' % (
            '零点已确认' if state.origin_confirmed else '待手动对零确认',
            '通信已验收' if self.machine.ready else '通信未就绪', len(self.bindings)))

    def refresh_ports(self):
        current = self.port_picker.currentText() or '/dev/ttyUSB0'
        self.port_picker.clear()
        self.port_picker.addItems(sorted({current} | {p.device for p in list_ports.comports()}))
        self.port_picker.setCurrentText(current)

    def _send(self, line):
        if self.transport is None:
            raise SerialTransportError('串口未连接')
        self.transport.write_line(line)
        self.log_view.append('HOST ' + line)
        self._persist_log('HOST ' + line)

    def connect_serial(self):
        if self._models_open() or self._closing:
            self.write_log('请先关闭模型管理窗口，再连接实机。')
            return
        if self.serial_port is not None:
            self.close_serial()
            return
        try:
            self.serial_port = self._serial_factory(self.port_picker.currentText(), 115200,
                                                    timeout=0, write_timeout=0.3)
            self.transport = SerialLineTransport(self.serial_port)
            self._handshake_at = time.monotonic() + 2
            self._flushed = False
            self.serial_button.setText('断开串口')
            self.write_log('串口已打开，等待 ESP32 启动后查询握手。')
        except Exception as error:
            self._finish_serial_close()
            self.write_log('连接失败：' + str(error))

    def close_serial(self):
        self._reset_deadline = None
        if self._disconnect_deadline is not None:
            return
        if self.serial_port is not None and self.machine.connected:
            self.machine.stop()
            self._disconnect_deadline = time.monotonic() + 30
            self.serial_button.setText('等待停机确认')
            return
        self._finish_serial_close()

    def _finish_serial_close(self):
        self._handshake_at = None
        self._disconnect_deadline = None
        self._reset_deadline = None
        self._job = None
        self._last_sort = None
        self.machine.disconnect()
        port, self.serial_port = self.serial_port, None
        self.transport = None
        if port is not None:
            try:
                port.close()
            except Exception as error:
                self.write_log('关闭串口：' + str(error))
        self.serial_button.setText('连接串口')
        self.update_state()
        if self._model_pending and self.vision is None:
            self._launch_models()
        if self._closing and self.vision is None:
            self._close_ready = True
            self.close()

    def poll_serial(self):
        if self.transport is not None:
            try:
                now = time.monotonic()
                if self._handshake_at is not None:
                    self.transport.read_lines()
                    if now >= self._handshake_at:
                        if not self._flushed:
                            self.serial_port.write(b'\n')
                            self._flushed = True
                            self._handshake_at = now + 0.3
                        else:
                            self.serial_port.reset_input_buffer()
                            self.transport = SerialLineTransport(self.serial_port)
                            self.machine.begin_handshake()
                            self._handshake_at = None
                            self._heartbeat_at = now
                else:
                    for line in self.transport.read_lines():
                        previous = self.machine.snapshot().completed_count
                        self.log_view.append('MCU ' + line)
                        self._persist_log('MCU ' + line)
                        handled = self.machine.handle_mcu(line)
                        if self.machine.snapshot().completed_count > previous:
                            self.record_completion()
                        stopped = handled and self.machine.snapshot().state == 'stopped' and (
                            line == 'STATUS,STOP' or line.startswith('STATUS,MCU,STOP,'))
                        if self._disconnect_deadline is not None and stopped:
                            self._finish_serial_close()
                            break
                        if self._reset_deadline is not None and stopped:
                            self._reset_deadline = None
                            epoch = self.machine.epoch
                            QTimer.singleShot(0, lambda epoch=epoch: self._confirm_reset(epoch))
                    if self.transport is not None and now - self._heartbeat_at >= 2:
                        self._send('STATUS?')
                        self._heartbeat_at = now
                self.machine.tick()
                if self._disconnect_deadline is not None and now >= self._disconnect_deadline:
                    self.write_log('未收到停机确认，现场需断开驱动电源。')
                    self._finish_serial_close()
                if self._reset_deadline is not None and now >= self._reset_deadline:
                    self._reset_deadline = None
                    self.write_log('复位取消：尚未收到真实停机确认。')
            except Exception as error:
                self._finish_serial_close()
                self.write_log('串口失效：' + str(error))
        if self.machine.snapshot().state in AUTO_STATES and not self._automatic_ready():
            self.stop()
        self._dispatch_vision()
        self.update_state()

    def connect_cameras(self):
        if self._models_open() or self._closing:
            self.write_log('请先关闭模型管理窗口，再连接相机。')
            return
        if self.vision is not None:
            self.stop()
            self.vision.shutdown()
            self.camera_button.setText('相机正在关闭')
            return
        try:
            self._frames_at = None
            self.vision = self._vision_factory(self.sources, self)
            self.vision.frames_ready.connect(self.on_frames)
            self.vision.recognition_ready.connect(self.on_recognized)
            self.vision.location_ready.connect(self.on_located)
            self.vision.failed.connect(self.on_vision_failed)
            self.vision.notice.connect(self.write_log)
            self.vision.finished.connect(self._vision_finished)
            self.vision.start()
            self.camera_button.setText('断开相机')
            self.write_log('正在打开三路真实相机。')
        except Exception as error:
            self.vision = None
            self.write_log('相机连接失败：' + str(error))

    def on_frames(self, frames):
        if self.vision is None or self._closing or self._model_pending:
            return
        self.frames = frames
        self._frames_at = time.monotonic()
        for role in ROLES:
            self.views[role].show_frame(frames[role])

    def _vision_finished(self):
        if self.vision is not None and not self.vision.wait(0):
            QTimer.singleShot(20, self._vision_finished)
            return
        worker, self.vision = self.vision, None
        if worker is not None:
            worker.deleteLater()
        self._job = None
        self._frames_at = None
        self.camera_button.setText('连接三路相机')
        for view in self.views.values():
            view.frame = None
            view.refresh()
        if self._model_pending and self.serial_port is None:
            self._launch_models()
        if self._closing and self.serial_port is None:
            self._close_ready = True
            self.close()

    def camera_settings(self):
        if self.vision is not None or self.machine.snapshot().state in AUTO_STATES:
            self.write_log('请先停机并断开相机，再修改端口。')
            return
        dialog = QDialog(self)
        dialog.setWindowTitle('相机角色与 USB 端口')
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        choices = {}
        devices = sorted(str(p) for p in Path('/dev/v4l/by-path').glob('*-video-index0'))
        for role, label in zip(ROLES, CAMERA_TITLES):
            combo = choices[role] = QComboBox()
            combo.setEditable(True)
            combo.addItems(sorted(set(devices + [self.sources[role]])))
            combo.setCurrentText(self.sources[role])
            form.addRow(label, combo)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QBUTTON_OK | QBUTTON_CANCEL)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() == QDIALOG_ACCEPTED:
            sources = {role: choices[role].currentText() for role in ROLES}
            if len({os.path.realpath(p) for p in sources.values()}) != 3:
                self.write_log('三个角色必须选择不同的相机。')
                return
            try:
                atomic_json(self.device_path, {'sources': sources})
                self.sources = sources
                self.calibration_confirmed.setChecked(False)
            except OSError as error:
                self.write_log('相机配置保存失败：' + str(error))

    def _command(self, command, description):
        try:
            accepted = command()
        except Exception as error:
            self.write_log(str(error))
            return False
        self.write_log(description if accepted else self.machine.snapshot().last_error or '操作未执行')
        return bool(accepted)

    def confirm_origin(self):
        if self._closing or self._models_open() or self._disconnect_deadline is not None:
            self.write_log('当前正在关闭或更新模型，无法确认原点。')
            return False
        if not self.machine.ready:
            self.write_log('请先连接串口并完成握手。')
            return False
        if self.machine.snapshot().state not in ('needs_origin', 'stopped') or self.machine.pending:
            self.write_log('请先停止设备并等待确认，再进行手动对零。')
            return False
        answer = QMessageBox.question(self, '确认手动零点',
            '没有原点限位开关。请确认机构已手动放到固定零点，工作范围已清空。\n'
            '确认后发送 RESET，启用电机并抬起推板，再等待 ESP32 确认。')
        yes = getattr(QMessageBox, 'Yes', None)
        if yes is None:
            yes = QMessageBox.StandardButton.Yes
        if answer != yes:
            return False
        return self._command(self.machine.confirm_origin, '零点确认已发送，等待设备回应。')

    def start(self):
        if (self.vision is None or self._frames_at is None or self._job is not None
                or time.monotonic() - self._frames_at > 2):
            self.write_log('需要三路新鲜画面且视觉任务空闲。')
            return False
        if not self.calibration_valid or not self.calibration_confirmed.isChecked():
            self.write_log('请先导入或保存九点标定，并逐点实测后勾选确认。')
            return False
        if not self._round_available():
            self.write_log(self._binding_error or '上一件投箱尚未确认，请核查六箱后清空本轮再开始。')
            return False
        if self._models_open() or self._disconnect_deadline is not None:
            self.write_log('模型管理或断开流程进行中。')
            return False
        return self._command(self.machine.start, '启动请求已发送。')

    def _automatic_ready(self):
        # A cold TensorRT initialization can temporarily pause previews. The
        # matching in-flight job remains bounded by LiveController's action
        # deadline; its final inference frames must still be read after loading.
        pending_capture = (self._job is not None and self._job[1] == self.machine.epoch
                           and self._job[2] in ('locate', 'sort')
                           and self.machine.snapshot().state in ('locating', 'recognizing'))
        return (self.machine.ready and self.vision is not None and self._frames_at is not None
                and (time.monotonic() - self._frames_at <= 10 or pending_capture)
                and self.calibration_valid and self.calibration_confirmed.isChecked()
                and self._round_available(in_flight=True) and not self._models_open()
                and self._disconnect_deadline is None)

    def stop(self):
        self._reset_deadline = None
        self._last_sort = None
        return self._command(self.machine.stop, '停止请求已发送，继续前需重新手动对零。')

    def reset(self):
        if not self.machine.ready or self._models_open() or self._closing or self._disconnect_deadline is not None:
            self.write_log('复位需要通信就绪且设备连接有效。')
            return False
        if not self.stop():
            return False
        self._reset_deadline = time.monotonic() + 30
        self.write_log('复位先等待停机确认，随后提示手动回零，再发送 RESET。')
        return True

    def _confirm_reset(self, epoch):
        if epoch == self.machine.epoch:
            self.confirm_origin()

    def _request(self, kind):
        if self.vision is None or self._frames_at is None or self._job is not None:
            self.write_log('相机未就绪或已有视觉任务。')
            return False
        self._sequence += 1
        token = self._sequence
        self._job = (token, self.machine.epoch, kind)
        try:
            if kind in ('locate', 'manual_locate'):
                if not self.calibration_valid:
                    raise ValueError('请先保存九点标定。')
                accepted = self.vision.locate(token, validate_points(self.calibration_rows), TRAY_ROI, (4, 0))
            else:
                accepted = self.vision.recognize(token, settle=True) if kind == 'sort' else self.vision.recognize(token)
            self.write_log('视觉请求 token=%s epoch=%s kind=%s' % (token, self.machine.epoch, kind))
        except Exception as error:
            self._job = None
            self.write_log('视觉请求失败：' + str(error))
            if kind in ('locate', 'sort'):
                self.stop()
            return False
        if not accepted:
            self._job = None
        return accepted

    def manual_recognize(self):
        if self.machine.snapshot().state in AUTO_STATES or self.machine.pending:
            self.write_log('请先停止自动流程，再手动识别。')
            return False
        return self._request('manual')

    def manual_locate(self):
        if self.machine.snapshot().state in AUTO_STATES or self.machine.pending:
            self.write_log('请先停止自动流程，再手动定位。')
            return False
        return self._request('manual_locate')

    def _dispatch_vision(self):
        if self._job is not None:
            return
        state = self.machine.snapshot().state
        if state in ('locating', 'recognizing'):
            if state == 'recognizing':
                if self._scan_epoch != self.machine.epoch:
                    self._scan_epoch = self.machine.epoch
                    self._scan_gate = ScanConsensus()
                    self._scan_next_at = 0
                if time.monotonic() < self._scan_next_at:
                    return
            if not self._request('locate' if state == 'locating' else 'sort'):
                self.stop()

    def _take_job(self, token):
        if self._job is None or self._job[0] != token:
            return None
        _, epoch, kind = self._job
        self._job = None
        return kind if epoch == self.machine.epoch else None

    def on_recognized(self, token, result):
        kind = self._take_job(token)
        if kind is None:
            self._persist_log('丢弃过期识别结果 token=%s epoch=%s' % (token, self.machine.epoch))
            return
        self._save_diagnostic(result, token, kind)
        if result.error:
            self.write_log('视觉失败：' + str(result.error))
            if kind == 'sort':
                self.stop()
            return
        detection, color = result.detection, result.color
        shape = detection.class_name if detection else '无目标'
        color_name = color.name if color else '未知颜色'
        confidence = '%.1f%%' % (detection.confidence * 100) if detection else '—'
        self.result_label.setText('%s / %s · %s · %.1f ms\n%s' % (
            shape, color_name, confidence, result.elapsed_ms, surface_summary(result)))
        sources = {'mask': '表面掩膜', 'box': '检测框（覆膜未参与）', 'none': '未取色'}
        color_sources = getattr(result, 'color_sources', ('none', 'none'))
        self.write_log(surface_summary(result) + '；取色 左：%s / 右：%s' % tuple(
            sources.get(source, source) for source in color_sources))
        self.write_log('识别结果 token=%s kind=%s shape=%s shape_conf=%s color=%s color_conf=%s pixels=%s elapsed_ms=%.1f' % (
            token, kind, shape, getattr(detection, 'confidence', None), color_name,
            getattr(color, 'confidence', None), getattr(color, 'valid_pixels', None), result.elapsed_ms))
        self.preview.show_frame(result.preview_bgr)
        if kind == 'sort':
            if not self._automatic_ready() or self.machine.snapshot().state != 'recognizing':
                self.stop()
                return
            usable = (detection is not None and color is not None and color.valid_pixels > 0
                      and np.isfinite(color.confidence) and color.confidence >= 0.35
                      and shape in self.names and color_name in COLORS)
            if self._scan_gate is None or self._scan_epoch != self.machine.epoch:
                self.write_log('缺少本件复核记录，未发送投箱指令。')
                self.stop()
                return
            decision = self._scan_gate.observe((shape, color_name) if usable else None)
            self.write_log('自动复核 %s/%s：%s / %s，连续一致 %s/%s' % (
                self._scan_gate.attempts, self._scan_gate.max_attempts, shape, color_name,
                self._scan_gate.matches, self._scan_gate.required_matches))
            if decision == 'retry':
                self._scan_next_at = time.monotonic() + .25
                return
            if decision == 'exhausted':
                self.stop()
                self.write_log('8 次复核未得到连续一致的有效形状和颜色，未分配箱号、未投箱。请导出诊断。')
                return
            try:
                # Persist the reservation before sending SORT. A missing DONE
                # must not let a different category reuse the same physical box.
                bin_id = self.auto_bins.reserve(shape, color_name)
            except CapacityFull:
                self.stop()
                self.write_log('检测到第七种新类别：六箱已绑定，已停机，本件未发送投箱指令。')
                return
            except (OSError, ValueError) as error:
                self.stop()
                self.write_log('自动分箱未执行：' + str(error))
                return
            self._refresh_binding_display()
            if self.machine.provide_sort(bin_id):
                self._last_sort = (shape, color_name, bin_id, result)
                self.write_log('%s %s → %s号箱，等待投箱完成。' % (color_name, shape, bin_id))
            else:
                self.stop()

    def on_located(self, token, target):
        kind = self._take_job(token)
        if kind is None:
            return
        self.write_log('托盘无目标' if target is None else '托盘目标 %.1f, %.1f mm' % target)
        if kind == 'locate':
            if not self._automatic_ready() or self.machine.snapshot().state != 'locating':
                self.stop()
                return
            accepted = self.machine.provide_target(None) if target is None else self.machine.provide_target(*target)
            if not accepted:
                self.stop()

    def on_vision_failed(self, token, message):
        job = self._job
        if token != -1 and (job is None or job[0] != token):
            self.write_log('视觉结果：' + str(message))
            return
        self._job = None
        self.write_log('视觉失败：' + str(message))
        if self.machine.connected and (token == -1 or (job and job[2] in ('locate', 'sort'))):
            self.stop()

    def record_completion(self):
        if self._last_sort is None:
            self.write_log('DONE 缺少对应识别记录，已停止。')
            self.stop()
            return
        shape, color, bin_id, result = self._last_sort
        try:
            self.auto_bins.complete(bin_id)
        except (OSError, ValueError) as error:
            self.stop()
            self.write_log('已收到 DONE，但本轮记录未确认保存：' + str(error))
            return
        self._last_sort = None
        total = self.auto_bins.total
        row = self.records.rowCount()
        self.records.insertRow(row)
        for column, value in ((0, total), (1, '%s %s → %s号盒' % (color, shape, bin_id)), (3, total)):
            self.records.setItem(row, column, QTableWidgetItem(str(value)))
        image = ImageView('')
        image.setMinimumHeight(0)
        image.setFixedHeight(64)
        image.show_frame(result.preview_bgr)
        self.records.setCellWidget(row, 2, image)
        self.records.setRowHeight(row, 68)
        if self.records.rowCount() > 200:
            self.records.removeRow(0)
        self._refresh_binding_display()
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            record = {'time': datetime.now().isoformat(), 'shape': shape, 'color': color,
                      'bin': bin_id, 'session_total': total, 'round_id': self.auto_bins.round_id}
            with (self.data_dir / 'live_results.jsonl').open('a', encoding='utf8') as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as error:
            self.stop()
            self.write_log('结果保存失败：' + str(error))

    @property
    def bindings(self):
        return self.auto_bins.bindings if self.auto_bins is not None else {}

    def _load_auto_bins(self):
        try:
            self.auto_bins = AutoBinSession(self.auto_bins_path, self.names, COLORS)
        except (OSError, ValueError) as error:
            self._binding_error = '本轮箱号记录无法读取，已禁止自动运行：' + str(error)
            self.write_log(self._binding_error)
        self._refresh_binding_display()
        if self.auto_bins is not None and self.auto_bins.pending is not None:
            self.write_log('存在未收到 DONE 确认的投箱，请检查六箱；清空本轮后才能重新运行。')

    def _round_available(self, in_flight=False):
        if self.auto_bins is None or self._binding_error is not None:
            return False
        pending = self.auto_bins.pending
        if pending is None:
            return True
        return bool(in_flight and self._last_sort is not None
                    and self.machine.snapshot().state == 'wait_done'
                    and (pending['shape'], pending['color'], pending['bin']) == self._last_sort[:3])

    def _refresh_binding_display(self):
        if self.auto_bins is None:
            self.total_label.setText('本轮记录待检查')
            for number, label in enumerate(self.bin_labels, 1):
                label.setText('%s号箱 · 待检查' % number)
                label.setToolTip(self._binding_error or '尚未读取本轮记录')
            return
        entries = {row['bin']: row for row in self.auto_bins.entries} if self.auto_bins else {}
        pending = self.auto_bins.pending if self.auto_bins else None
        total = self.auto_bins.total if self.auto_bins else 0
        self.total_label.setText('本轮成功总数 %s' % total)
        for number, label in enumerate(self.bin_labels, 1):
            row = entries.get(number)
            label.setText('%s号箱 · %s件' % (number, row['count'] if row else 0))
            label.setToolTip('%s %s' % (row['color'], row['shape']) if row else '按首次出现顺序自动绑定')
            if row:
                label.setText(label.text() + '\n%s %s' % (row['color'], row['shape']))
            if pending is not None and pending['bin'] == number:
                label.setText(label.text() + ' · 待确认')

    def clear_round(self):
        if (self.machine.snapshot().state in AUTO_STATES or self.machine.pending or self._job is not None
                or self._closing or self._models_open() or self._disconnect_deadline is not None):
            self.write_log('请先停机并等待当前动作结束，再清空本轮。')
            return False
        if (self.machine.connected or self.serial_port is not None) and not self.machine.idle_confirmed:
            self.write_log('尚未收到设备空闲或停机确认，暂不能清空本轮。')
            return False
        message = '请确认六个箱子里的物块都已取出。'
        if not self.machine.connected and self.serial_port is None:
            message = ('当前未连接设备，无法确认机构已经停止。\n'
                       '请先断开电机和舵机驱动供电，确认机构停止，并清空六个箱子后再确认。')
        message += '\n清空本轮会重置自动箱号和本轮计数，下一种有效类别从1号箱开始。'
        answer = QMessageBox.question(self, '开始新一轮', message)
        yes = getattr(QMessageBox, 'Yes', None)
        if yes is None:
            yes = QMessageBox.StandardButton.Yes
        if answer != yes:
            return False
        try:
            session = self.auto_bins or AutoBinSession(self.auto_bins_path, self.names, COLORS, load_existing=False)
            session.clear()
        except (OSError, ValueError) as error:
            self.write_log('清空本轮失败：' + str(error))
            return False
        self.auto_bins = session
        self._binding_error = None
        self._last_sort = None
        self.records.setRowCount(0)
        self._refresh_binding_display()
        self.write_log('已开始新一轮，按形状＋颜色首次出现顺序自动分配1至6号箱。')
        return True

    def binding_dialog(self):
        dialog = self.create_binding_dialog()
        dialog.exec()
        dialog.deleteLater()

    def create_binding_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle('本轮自动箱号')
        dialog.resize(620, 330)
        layout = QVBoxLayout(dialog)
        hint = QLabel('形状＋颜色相同为同类；新类别按首次出现顺序绑定1至6号箱，第七类停机。')
        hint.setWordWrap(True)
        layout.addWidget(hint)
        table = QTableWidget(6, 4)
        table.setObjectName('auto_bin_table')
        table.setHorizontalHeaderLabels(('箱号', '形状类别', '颜色', '已完成数量'))
        no_edit = getattr(QAbstractItemView, 'NoEditTriggers', None)
        if no_edit is None:
            no_edit = QAbstractItemView.EditTrigger.NoEditTriggers
        table.setEditTriggers(no_edit)
        entries = {row['bin']: row for row in self.auto_bins.entries} if self.auto_bins else {}
        for number in range(1, 7):
            row = entries.get(number)
            for column, value in enumerate((number, row['shape'] if row else '未绑定',
                                             row['color'] if row else '—', row['count'] if row else 0)):
                table.setItem(number - 1, column, QTableWidgetItem(str(value)))
        table.setColumnWidth(0, 60)
        table.setColumnWidth(2, 80)
        table.setColumnWidth(3, 100)
        stretch = getattr(QHeaderView, 'Stretch', None)
        if stretch is None:
            stretch = QHeaderView.ResizeMode.Stretch
        table.horizontalHeader().setSectionResizeMode(1, stretch)
        layout.addWidget(table)
        if self.auto_bins is not None and self.auto_bins.pending is not None:
            layout.addWidget(QLabel('存在待确认投箱；计数仅在收到真实 DONE 后增加。'))
        layout.addWidget(button('关闭', dialog.accept))
        return dialog

    def create_calibration_dialog(self):
        return CalibrationDialog(self, self.calibration_rows)

    def show_calibration(self):
        if self.machine.snapshot().state in AUTO_STATES:
            self.write_log('请先停机，再进入标定。')
            return
        dialog = self.create_calibration_dialog()
        dialog.exec()
        dialog.deleteLater()

    def save_calibration(self, rows, frame=None):
        validate_points(rows)
        frame = normalize_frame_settings(self.calibration_frame if frame is None else frame)
        atomic_json(self.calibration_path, {'mode': 'live', 'points': rows, 'frame': frame})
        self.calibration_rows = rows
        self.calibration_frame = frame
        self.calibration_valid = True
        self.calibration_confirmed.setChecked(False)

    def _load_settings(self):
        try:
            if self.device_path.exists():
                sources = json.loads(self.device_path.read_text(encoding='utf8'))['sources']
                if set(sources) != set(ROLES) or len({os.path.realpath(p) for p in sources.values()}) != 3:
                    raise ValueError('相机角色配置不合法')
                self.sources = sources
            if self.calibration_path.exists():
                saved = json.loads(self.calibration_path.read_text(encoding='utf8'))
                rows = saved['points']
                if saved['mode'] != 'live' or len(rows) != 9 or {r[0] for r in rows} != {'P%s' % n for n in range(1, 10)}:
                    raise ValueError('实机九点标定格式不正确')
                validate_points(rows)
                self.calibration_frame = normalize_frame_settings(saved.get('frame'))
                self.calibration_rows = rows
                self.calibration_valid = True
        except (KeyError, TypeError, ValueError, OSError) as error:
            self.write_log('配置未加载：' + str(error))

    def open_models(self):
        if self._models_open():
            self.write_log('模型验证窗口已经打开。')
            return
        self._model_pending = True
        self.close_serial()
        if self.vision is not None:
            self.vision.shutdown()
        elif self.serial_port is None and self._model_pending:
            self._launch_models()

    def _models_open(self):
        return self._model_pending or (self._model_process is not None and self._model_process.poll() is None)

    def _launch_models(self):
        self._model_pending = False
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            path = self.data_dir / 'model_trial_raw.png'
            from legacy_vision import prepare_legacy_pair
            raw, _, _ = prepare_legacy_pair(self.frames['left'], self.frames['right'])
            if not cv2.imwrite(str(path), raw):
                raise OSError('无法保存模型试跑图片')
            self._model_process = subprocess.Popen([sys.executable, str(ROOT / 'app.py'), '--raw', '--file', str(path)], cwd=str(ROOT))
            self.write_log('已打开离线模型验证窗口，通过「设置 / 日志 → 模型管理」更换模型。')
        except (OSError, ValueError) as error:
            self.write_log('模型窗口启动失败：' + str(error))

    def show_competition(self):
        self.pages.setCurrentWidget(self.competition_page)

    def show_debug(self):
        self.pages.setCurrentWidget(self.debug_page)

    def closeEvent(self, event):
        if not self._closing:
            self._closing = True
            self._model_pending = False
            self.close_serial()
            if self.vision is not None:
                self.vision.shutdown()
            elif self.serial_port is None:
                self._close_ready = True
        if not self._close_ready:
            event.ignore()
            return
        self.timer.stop()
        super().closeEvent(event)


def main():
    parser = argparse.ArgumentParser(description='Xavier NX 实机调试、比赛与九点标定')
    parser.add_argument('--fullscreen', action='store_true')
    args = parser.parse_args()
    application = QApplication(sys.argv[:1])
    window = LiveWindow()
    window.showFullScreen() if args.fullscreen else window.show()
    return application.exec()


if __name__ == '__main__':
    raise SystemExit(main())
