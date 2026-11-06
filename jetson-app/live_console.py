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
    from PySide6.QtWidgets import QCheckBox, QHeaderView, QStackedWidget, QTableWidget, QTableWidgetItem, QTextEdit
except ImportError:
    from PyQt5.QtWidgets import QCheckBox, QHeaderView, QStackedWidget, QTableWidget, QTableWidgetItem, QTextEdit

import cv2
import numpy as np
import serial
from serial.tools import list_ports

from esp32_serial import SerialLineTransport, SerialTransportError
from live_control import LiveController
from live_vision import LiveVisionWorker
from legacy_vision import surface_summary
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
        if not (0 <= px < 640 and 0 <= py < 480 and 0 <= x <= 160 and 0 <= y <= 160):
            raise ValueError('标定点超出画幅或当前 160mm 行程范围')
    return fit_calibration(rows)


class LiveWindow(QMainWindow):
    simulation_mode = False

    def __init__(self, data_dir=None, vision_factory=LiveVisionWorker, serial_factory=serial.Serial):
        super().__init__()
        self.data_dir = Path(data_dir) if data_dir else ROOT / 'data'
        self.calibration_path = self.data_dir / 'live_calibration.json'
        self.bindings_path = self.data_dir / 'live_bindings.json'
        self.device_path = self.data_dir / 'live_devices.json'
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
        self.bindings = {}
        self.calibration_valid = False
        self.calibration_rows = list(DEFAULT_POINTS)
        self.setWindowTitle('YLERS · Xavier NX 实机分拣操作台')
        self.setStyleSheet(STYLE)
        self.resize(1024, 680)
        self._make_ui()
        self._load_settings()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll_serial)
        self.timer.start(40)
        self.write_log('实机入口已打开。先连接相机与串口，手动对零后确认。')

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
            ('箱号绑定', self.binding_dialog), ('运行', self.start), ('停止', self.stop),
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
        self.total_label = QLabel('本次成功总数 0')
        self.total_label.setStyleSheet('font-size:24px;font-weight:bold;color:#147d73')
        layout.addWidget(self.total_label)
        self.records = QTableWidget(0, 4)
        self.records.setHorizontalHeaderLabels(('序号', '货物名称', '图片', '本次成功总数'))
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
            self.bin_labels.append(label)
            boxes.addWidget(label)
        layout.addLayout(boxes)

    def write_log(self, message):
        line = datetime.now().strftime('%H:%M:%S') + ' ' + str(message)
        self.log_view.append(line)
        self.statusBar().showMessage(str(message))
        self.update_state()

    def update_state(self):
        state = self.machine.snapshot()
        self.machine_status.setText('状态 %s · XY %.1f, %.1f mm · 升降 %s · %s' % (
            state.state, state.x_mm, state.y_mm, '下降' if state.pusher_down else '抬起',
            '动作等待确认' if self.machine.pending else ''))
        self.banner.setText('实机模式 · %s · %s' % (
            '零点已确认' if state.origin_confirmed else '待手动对零确认',
            '通信已验收' if self.machine.ready else '通信未就绪'))

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
        if not self.bindings:
            self.write_log('请先配置箱号与形状、颜色的绑定。')
            return False
        if self._models_open() or self._disconnect_deadline is not None:
            self.write_log('模型管理或断开流程进行中。')
            return False
        return self._command(self.machine.start, '启动请求已发送。')

    def _automatic_ready(self):
        return (self.machine.ready and self.vision is not None and self._frames_at is not None
                and time.monotonic() - self._frames_at <= 10
                and self.calibration_valid and self.calibration_confirmed.isChecked()
                and bool(self.bindings) and not self._models_open()
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
                accepted = self.vision.recognize(token)
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
            return
        if result.error:
            self.on_vision_failed(token, result.error)
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
        self.preview.show_frame(result.preview_bgr)
        if kind == 'sort':
            if not self._automatic_ready() or self.machine.snapshot().state != 'recognizing':
                self.stop()
                return
            usable = (detection is not None and color is not None and color.valid_pixels > 0
                      and color.confidence >= 0.35)
            bin_id = self.bindings.get((shape, color_name)) if usable else None
            if bin_id is None:
                self.write_log('识别不确定或没有对应箱号，流程已停止。')
                self.stop()
            elif self.machine.provide_sort(bin_id):
                self._last_sort = (shape, color_name, bin_id, result)
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
        state = self.machine.snapshot()
        if self._last_sort is None:
            self.write_log('DONE 缺少对应识别记录，已停止。')
            self.stop()
            return
        shape, color, bin_id, result = self._last_sort
        self._last_sort = None
        total = state.completed_count
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
        self.total_label.setText('本次成功总数 %s' % total)
        for index, label in enumerate(self.bin_labels):
            label.setText('%s号盒 · %s' % (index + 1, state.bin_counts[index]))
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            record = {'time': datetime.now().isoformat(), 'shape': shape, 'color': color,
                      'bin': bin_id, 'session_total': total}
            with (self.data_dir / 'live_results.jsonl').open('a', encoding='utf8') as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as error:
            self.stop()
            self.write_log('结果保存失败：' + str(error))

    def save_bindings(self, rows):
        bindings = {}
        bins = set()
        for row in rows:
            shape, color, number = row['shape'], row['color'], row['bin']
            if (shape not in self.names or color not in COLORS or type(number) is not int
                    or not 1 <= number <= 6 or number in bins or (shape, color) in bindings):
                raise ValueError('箱号绑定包含未知类别、颜色、重复项或非法箱号')
            bins.add(number)
            bindings[(shape, color)] = number
        atomic_json(self.bindings_path, rows)
        self.bindings = bindings

    def binding_dialog(self):
        if self.machine.snapshot().state in AUTO_STATES:
            self.write_log('请先停机再修改箱号。')
            return
        dialog = QDialog(self)
        dialog.setWindowTitle('六箱绑定')
        layout = QVBoxLayout(dialog)
        table = QTableWidget(6, 3)
        table.setHorizontalHeaderLabels(('箱号', '形状类别', '颜色'))
        choices = []
        for index in range(6):
            table.setItem(index, 0, QTableWidgetItem(str(index + 1)))
            shape, color = QComboBox(), QComboBox()
            shape.addItems(['未绑定'] + self.names)
            color.addItems(COLORS)
            for (name, color_name), number in self.bindings.items():
                if number == index + 1:
                    shape.setCurrentText(name)
                    color.setCurrentText(color_name)
            table.setCellWidget(index, 1, shape)
            table.setCellWidget(index, 2, color)
            choices.append((shape, color))
        layout.addWidget(table)
        buttons = QDialogButtonBox(QBUTTON_OK | QBUTTON_CANCEL)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() == QDIALOG_ACCEPTED:
            rows = [{'shape': shape.currentText(), 'color': color.currentText(), 'bin': index + 1}
                    for index, (shape, color) in enumerate(choices) if shape.currentIndex() > 0]
            try:
                self.save_bindings(rows)
                self.write_log('已保存箱号绑定。')
            except (OSError, ValueError) as error:
                self.write_log(str(error))

    def create_calibration_dialog(self):
        return CalibrationDialog(self, self.calibration_rows)

    def show_calibration(self):
        if self.machine.snapshot().state in AUTO_STATES:
            self.write_log('请先停机，再进入标定。')
            return
        dialog = self.create_calibration_dialog()
        dialog.exec()
        dialog.deleteLater()

    def save_calibration(self, rows):
        validate_points(rows)
        atomic_json(self.calibration_path, {'mode': 'live', 'points': rows})
        self.calibration_rows = rows
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
                self.calibration_rows = rows
                self.calibration_valid = True
            if self.bindings_path.exists():
                self.save_bindings(json.loads(self.bindings_path.read_text(encoding='utf8')))
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
