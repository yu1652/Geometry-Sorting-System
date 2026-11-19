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
from types import SimpleNamespace

from app import exec_qt
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
from live_control import AUTO_STATES, LiveController
from live_vision import LiveVisionWorker
from legacy_vision import surface_summary
from calibration_frame import normalize_frame_settings
from motion_limits import MACHINE_REGION, X_MAX_MM, Y_MAX_MM, position_in_machine
from auto_bins import AutoBinSession, CapacityFull
from runtime_diagnostics import RuntimeDiagnostics
from scan_gate import ScanConsensus
from sorter_simulation import PARK_MM, PUSH_END_X_MM
import linux_20261107 as release
from sorter_console import (CalibrationDialog, DEFAULT_POINTS, TRAY_ROI,
                            button, fit_calibration)

ROOT = Path(__file__).resolve().parent
ROLES = ('left', 'right', 'top')
CAMERA_TITLES = ('识别区俯拍 / 拼图左', '识别区侧视 / 拼图右', '托盘定位俯拍')
COLORS = ('红色', '橙色', '黄色', '绿色', '青色', '蓝色', '紫色', '黑色', '白色', '灰色')
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


def validate_points(rows, *, roi=TRAY_ROI, physical_region=MACHINE_REGION):
    if (not isinstance(rows, (list, tuple)) or len(rows) != 9
            or any(len(row) != 5 for row in rows)
            or {row[0] for row in rows} != {'P%s' % n for n in range(1, 10)}):
        raise ValueError('需要 P1 至 P9 各一个完整标定点')
    for _, px, py, x, y in rows:
        if not (0 <= px < 640 and 0 <= py < 480 and position_in_machine(x, y)):
            raise ValueError(f'标定点超出画幅或机械行程：X 0–{X_MAX_MM:g}、Y 0–{Y_MAX_MM:g} mm')
    return fit_calibration(rows, roi=roi, physical_region=physical_region)


def _system_boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text(encoding='ascii').strip()


def _claim_fixed_origin_boot():
    try:
        boot_id = _system_boot_id()
        marker = Path(tempfile.gettempdir()) / ('ylers-fixed-origin-%d-%s' % (os.getuid(), boot_id))
        marker.touch(mode=0o600, exist_ok=False)
    except OSError:
        return False
    return True


class LiveWindow(QMainWindow):
    simulation_mode = False

    def __init__(self, data_dir=None, vision_factory=LiveVisionWorker, serial_factory=serial.Serial, *,
                 linux_20261107=False, butu_24=False, startup_serial_port=None, fixed_origin_on_boot=False,
                 round_boot_id=None):
        super().__init__()
        self.butu_24 = butu_24
        if butu_24:
            import butu_20261110 as profile
        else:
            profile = release if linux_20261107 else None
        from side_model_store import SideModelStore, LiveSideProfile
        self._active_side_model = SideModelStore(ROOT).read_active()
        if self._active_side_model is not None:
            if profile is None:
                import butu_20261110 as profile
            profile = LiveSideProfile(ROOT, profile, self._active_side_model)
        self._shape_profile = profile
        self._profile_label = '补图 24 类' if butu_24 else 'Linux 20261107'
        if self._active_side_model is not None:
            self._profile_label = '已导入 %d 类 · 320 · %s' % (len(profile.CLASS_NAMES),
                '特征输入' if self._active_side_model['preprocess'] == 'legacy_features' else 'RGB输入')
        linux_20261107 = self.linux_20261107 = profile is not None
        self._models_warm = False
        self.release_competition = None
        self._round_images = {}
        self._current_item = {}
        self._tray_status = '等待托盘定位'
        self._release_status = None
        self._competition_preview_at = 0.0
        self.data_dir = Path(data_dir) if data_dir else ROOT / 'data'
        self.calibration_path = self.data_dir / 'live_calibration.json'
        self.tray_background_path = self.data_dir / 'tray_background.npz'
        self._tray_background_capture = None
        self._tray_location_hold = None
        self._tray_location_overlay = None
        round_suffix = '_butu24' if butu_24 else ('_20261107' if linux_20261107 else '')
        if self._active_side_model is not None:
            round_suffix = '_live_side_' + self._active_side_model['engine_sha256'][:12]
        round_dir = self.data_dir / 'boot_rounds' / round_boot_id if round_boot_id else self.data_dir
        self.auto_bins_path = round_dir / ('live_auto_bins' + round_suffix + '.json')
        self.device_path = self.data_dir / 'live_devices.json'
        self.diagnostics = RuntimeDiagnostics(self.data_dir)
        self._diagnostic_log_error = False
        self._scan_gate = None
        self._scan_epoch = None
        self._scan_next_at = 0
        self.calibration_frame = normalize_frame_settings()
        self.names = (profile.CLASS_NAMES if profile is not None else
                      (ROOT / 'models/legacy_classes.txt').read_text(encoding='utf8').splitlines())
        self.sources = dict(DEFAULT_SOURCES)
        self._vision_factory, self._serial_factory = vision_factory, serial_factory
        self.machine = LiveController(self._send, verify_drop=False, accept_boot_origin=True,
                                      scan_timeout=release.SCAN_TIMEOUT if linux_20261107 else None,
                                      done_timeout=release.DONE_TIMEOUT if linux_20261107 else None)
        self.serial_port = self.transport = self.vision = None
        self._startup_serial_port = startup_serial_port
        self._fixed_origin_on_boot = fixed_origin_on_boot
        self._startup_zero_reset_pending = False
        self._startup_connections = set()
        self._startup_timer = QTimer(self)
        self._startup_timer.timeout.connect(self._connect_startup_devices)
        self._handshake_at = None
        self._flushed = False
        self._previous_mcu_line = None
        self._disconnect_deadline = None
        self._reset_deadline = None
        self._reset_action = None
        self._reset_generation_seen = 0
        self._heartbeat_at = time.monotonic()
        self._closing = False
        self._close_ready = False
        self._sequence = 0
        self._job = None
        self._last_sort = None
        self._model_pending = False
        self._model_process = None
        self._model_restart_required = False
        self._restore_fullscreen = False
        self._frames_at = None
        self.frames = {role: np.zeros((480, 640, 3), np.uint8) for role in ROLES}
        self.auto_bins = None
        self._binding_error = None
        self.calibration_valid = False
        self.calibration_rows = list(DEFAULT_POINTS)
        self.setWindowTitle('YLERS · Xavier NX 实机分拣操作台' + (' · ' + self._profile_label if linux_20261107 else ''))
        self.setStyleSheet(STYLE)
        self.resize(1024, 600 if linux_20261107 else 680)
        self._make_ui()
        self.calibration_confirmed.setText('九点标定数据已校验（自动加载或保存）')
        self.calibration_confirmed.setEnabled(False)
        self._load_settings()
        self._load_auto_bins()
        if self.linux_20261107:
            self.show_competition()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll_serial)
        self.timer.start(40)
        self.write_log('实机入口已打开。连接相机与串口；ESP32 启动位置为本次软件零点，握手校验后自动接受。')
        if round_boot_id and self.auto_bins is not None and not self.auto_bins.entries:
            self.write_log('本次开机的六个箱号均待绑定，计数为 0。')
        if self.butu_24 or self._active_side_model is not None:
            self.write_log('表面掩膜已停用，不加载掩膜 Engine；颜色使用检测框 HSV 取样，形状特征预处理保留。')
        if self._active_side_model is not None:
            self.write_log('正式上位机形状模型：%s；Engine=%s；SHA256=%s' %
                (self._profile_label, self._active_side_model['engine_path'], self._active_side_model['engine_sha256']))
        attempts = release.SCAN_ATTEMPTS if linux_20261107 else 8
        self.write_log('自动识别：模型就绪后重新取图，等待至少 0.8 秒；最多复核 %d 次，连续两次形状和颜色一致才投箱。' % attempts)
        if linux_20261107:
            self.write_log('%s：%d 类形状，收到投箱完成回执后计数，本轮目标 %d 件。'
                           % (self._profile_label, len(self.names), release.ROUND_SIZE))
        self.write_log('识别仅依据形状和颜色模型；无有效目标时有限重试，耗尽后停止，不投箱、不计数。')
        if startup_serial_port:
            self._startup_connections.update(('serial', 'cameras'))
            self._startup_deadline = time.monotonic() + 20
            self._startup_timer.start(500)

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
        model_text = '模型管理' if self._active_side_model is None else '模型管理·%d类' % len(self.names)
        self.model_button = button(model_text, self.open_models)
        if self._active_side_model is not None:
            self.model_button.setToolTip(self._active_side_model['engine_path'])
        connection.addWidget(self.model_button)
        self.diagnostic_button = button('导出诊断', self.export_diagnostics)
        connection.addWidget(self.diagnostic_button)
        outer.addLayout(connection)
        self.pages = QStackedWidget()
        outer.addWidget(self.pages, 1)
        self.debug_page, self.competition_page = QWidget(), QWidget()
        self.pages.addWidget(self.debug_page)
        self.pages.addWidget(self.competition_page)
        self.machine_status = QLabel('串口未连接')
        if self.linux_20261107:
            from release_competition import ReleaseCompetition
            from release_debug import build_release_debug

            self.banner.hide()
            connection.setSpacing(4)
            root.setStyleSheet(
                'QPushButton {padding:3px 7px;min-height:18px;}'
                'QComboBox {padding:2px 5px;}'
            )
            build_release_debug(self)
            self.debug_park_label.setText('PARK 等待点\n%g, %g mm' % PARK_MM)
            self.debug_push_end_label.setText('推料终止线\nX=%g mm' % PUSH_END_X_MM)
            competition = QVBoxLayout(self.competition_page)
            competition.setContentsMargins(0, 0, 0, 0)
            self.release_competition = ReleaseCompetition(self)
            competition.addWidget(self.release_competition)
            return

        self._make_debug_page()
        self.tray_log_view = self.serial_log_view = self.log_view
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
            control = self.actions[label] = button('完全复位' if label == '复位' else label, callback)
            controls.addWidget(control)
        outer.addLayout(controls)
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

    def write_log(self, message, *, view=None):
        line = datetime.now().strftime('%H:%M:%S') + ' ' + str(message)
        (self.log_view if view is None else view).append(line)
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
                              ROOT / 'data/model_store/active.json', self.tray_background_path),
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
        self.banner.setText((self._profile_label if self.linux_20261107 else '实机模式') + ' · %s · %s · 自动分箱 %s/6' % (
            '零点已确认' if state.origin_confirmed else '待手动对零确认',
            '通信已验收' if self.machine.ready else '通信未就绪', len(self.bindings)))
        if self.release_competition is not None:
            status = self._release_status_values(state)
            self.debug_status_label.setText('状态：' + status[0])
            self.debug_serial_label.setText('串口：' + status[1])
            self.debug_tray_status_label.setText('托盘定位：' + status[2])
            self.debug_position_label.setText(self.machine_status.text())
            self.debug_home_label.setText('HOME 原点\n' + ('0, 0 mm' if state.origin_confirmed else '待对零'))
            paused = self.machine.feed_pause_requested
            for button in (self.debug_pause_button, self.release_competition.pause_button):
                button.setText('继续' if paused else '暂停')
                button.setEnabled(self.machine.ready and state.state in AUTO_STATES)
            if status != self._release_status:
                self._refresh_binding_display()

    def _release_status_values(self, state):
        if self.machine.reset_in_progress:
            status = '完全复位中，等待 HOME 和控制板回执'
        elif self._model_restart_required:
            status = '模型已更新，请退出并重启上位机'
        elif self._binding_error:
            status = '本轮记录待检查'
        elif self.auto_bins is not None and self.auto_bins.total >= release.ROUND_SIZE:
            status = '本轮已完成'
        elif state.state.startswith('reject_'):
            status = ('未知料排出中，关板后暂停进料' if self.machine.feed_pause_requested
                      else '未知料排出中，关板后继续进料')
        elif state.state == 'paused':
            status = '已暂停进料'
        elif self.machine.feed_pause_requested:
            status = '本件完成后暂停进料'
        elif state.state in ('enabling_auto', 'starting'):
            status = '启动中，等待控制板确认'
        elif state.state in AUTO_STATES:
            status = '自动分拣中'
        elif state.state == 'armed':
            status = self._start_block_reason(state)
            if status is None:
                status = '设备就绪，可按实体启动键' if self.machine.idle_confirmed else '等待控制板待机'
        elif state.state == 'stopped':
            status = '已停止，启动前确认零点是否改变' if self.machine.can_reuse_origin else '已停止，坐标记录待恢复'
        elif self.machine.ready:
            status = '坐标记录待确认'
        else:
            status = '等待设备就绪'
        serial_status = '已验收' if self.machine.ready else ('握手中' if self.serial_port else '未连接')
        tray_status = self._tray_status if self._frames_at is not None else '相机未就绪'
        return status, serial_status, tray_status

    def refresh_ports(self):
        current = self.port_picker.currentText() or '/dev/ttyUSB0'
        self.port_picker.clear()
        self.port_picker.addItems(sorted({current} | {p.device for p in list_ports.comports()}))
        self.port_picker.setCurrentText(current)

    def _connect_startup_devices(self):
        pending = self._startup_connections
        if self._closing or self._models_open():
            pending.clear()
        if 'serial' in pending and os.path.exists(self._startup_serial_port):
            pending.remove('serial')
            self.port_picker.setCurrentText(self._startup_serial_port)
            self.connect_serial()
            self._startup_zero_reset_pending = self._fixed_origin_on_boot and self.serial_port is not None
        if 'cameras' in pending and all(os.path.exists(path) for path in self.sources.values()):
            pending.remove('cameras')
            self.connect_cameras()
        if pending and time.monotonic() >= self._startup_deadline:
            missing = []
            if 'serial' in pending:
                missing.append('串口 ' + self._startup_serial_port)
            if 'cameras' in pending:
                missing.extend('%s %s' % (title, self.sources[role])
                               for role, title in zip(ROLES, CAMERA_TITLES)
                               if not os.path.exists(self.sources[role]))
            self.write_log('开机连接等待超时：' + '；'.join(missing))
            pending.clear()
        if not pending:
            self._startup_timer.stop()

    def _send(self, line):
        if self.transport is None:
            raise SerialTransportError('串口未连接')
        self.transport.write_line(line)
        self.serial_log_view.append('HOST ' + line)
        self._persist_log('HOST ' + line)

    def connect_serial(self):
        if self._models_open() or self._closing:
            self.write_log('请先关闭模型管理窗口，再连接实机。')
            return
        self._startup_connections.discard('serial')
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
        self._tray_background_capture = self._tray_location_hold = None
        self._tray_location_overlay = None
        self._reset_action = None
        self._handshake_at = None
        self._previous_mcu_line = None
        self._startup_zero_reset_pending = False
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
                        self.serial_log_view.append('MCU ' + line)
                        self._persist_log('MCU ' + line)
                        handled = self.machine.handle_mcu(line)
                        state = self.machine.snapshot()
                        if handled and not self.machine.connected:
                            self._finish_serial_close()
                            self.write_log(state.last_error or '串口控制已断开。', view=self.serial_log_view)
                            break
                        if state.completed_count > previous:
                            self.record_completion()
                        self._handle_start_button(line, state)
                        stopped = handled and self.machine.snapshot().state == 'stopped' and (
                            line == 'STATUS,STOP' or line.startswith('STATUS,MCU,STOP,'))
                        if self._disconnect_deadline is not None and stopped:
                            self._finish_serial_close()
                            break
                        self._finish_reset_action()
                    if self.transport is not None and now - self._heartbeat_at >= 2:
                        self._send('STATUS?')
                        self._heartbeat_at = now
                self._confirm_fixed_origin_at_startup()
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

    def _confirm_fixed_origin_at_startup(self):
        if not self._startup_zero_reset_pending or not self.machine.ready:
            return
        self._startup_zero_reset_pending = False
        state = self.machine.snapshot()
        if state.origin_confirmed:
            self.write_log('开机零点已确认，继续检查相机与模型就绪。')
        elif (self.calibration_valid and state.state == 'needs_origin' and self.machine.idle_confirmed
              and (state.x_mm, state.y_mm) == (0.0, 0.0)):
            self._command(self.machine.confirm_origin, '开机检测到控制板停机且坐标为 (0,0)，等待零点确认回执。')
        else:
            self.write_log('开机无法自动确认零点，请核对控制板状态、坐标和九点标定。')

    def _handle_start_button(self, line, state):
        previous, self._previous_mcu_line = self._previous_mcu_line, line
        # The existing firmware emits WAIT_HOST both on a button press and on a
        # rejected serial START. Only the latter has an immediate RX,START echo.
        if line != 'STATUS,WAIT_HOST' or previous == 'RX,START':
            return
        if state.state == 'armed' and state.origin_confirmed and self.machine.idle_confirmed:
            self.write_log('已收到实体启动键，正在检查设备就绪状态。')
            if self.start():
                self.write_log('已接受实体启动键。')
        elif state.state in ('needs_origin', 'stopped'):
            self.write_log('已收到实体启动键，请先在界面确认有效零点。')

    def connect_cameras(self):
        if self._models_open() or self._closing:
            self.write_log('请先关闭模型管理窗口，再连接相机。')
            return
        self._startup_connections.discard('cameras')
        if self.vision is not None:
            self.stop()
            self.vision.shutdown()
            self.camera_button.setText('相机正在关闭')
            return
        try:
            self._frames_at = None
            self._models_warm = False
            settings = {}
            if self.linux_20261107:
                settings = dict(recognizer_factory=self._shape_profile.make_recognizer, tray_factory=release.make_tray_backend,
                                drain_batches=5, tray_feature_input=True,
                                tray_compensation=release.TRAY_COMPENSATION, warm_models=True,
                                tray_contact_ignore_unresolved=release.TRAY_CONTACT_IGNORE_UNRESOLVED,
                                tray_contact_side_clearance_mm=release.TRAY_CONTACT_SIDE_CLEARANCE_MM,
                                tray_contact_allow_sweep_mm2=release.TRAY_CONTACT_ALLOW_SWEEP_MM2)
            self.vision = self._vision_factory(self.sources, self, **settings)
            self.vision.frames_ready.connect(self.on_frames)
            self.vision.recognition_ready.connect(self.on_recognized)
            self.vision.location_ready.connect(self.on_located)
            self.vision.location_evidence_ready.connect(self.on_location_evidence)
            self.vision.tray_background_ready.connect(self.on_tray_background_ready)
            self.vision.failed.connect(self.on_vision_failed)
            self.vision.notice.connect(self.write_log)
            self.vision.models_ready.connect(self.on_models_ready)
            self.vision.finished.connect(self._vision_finished)
            self.vision.start()
            if self.linux_20261107 and not self.tray_background_path.is_file():
                self.write_log('尚未保存托盘底板，当前使用模型定位，可能漏掉边缘物块；清空后可在“更多”保存底板。')
            self.camera_button.setText('断开相机')
            self.write_log('正在打开三路真实相机。')
        except Exception as error:
            self.vision = None
            self.write_log('相机连接失败：' + str(error))

    def on_models_ready(self):
        if self.vision is not None and not self._closing:
            self._models_warm = True
            self.write_log('形状识别和托盘定位模型已试跑。')

    def on_frames(self, frames):
        if self.vision is None or self._closing or self._model_pending:
            return
        self.frames = frames
        self._frames_at = time.monotonic()
        if self.release_competition is None or self.pages.currentWidget() is self.debug_page:
            self._show_debug_frames()
        elif self._frames_at - self._competition_preview_at >= .15:
            self._show_competition_frames()

    def _show_debug_frames(self):
        for role in ROLES:
            frame = self.frames[role]
            if role == 'top' and self.release_competition is not None:
                frame = frame.copy()
                x, y, width, height = (release.TRAY_FOREGROUND_ROI if self.tray_background_path.is_file()
                                       else release.TRAY_ROI)
                cv2.rectangle(frame, (x, y), (x + width - 1, y + height - 1), (0, 220, 140), 1)
                if self.calibration_valid:
                    for name, px, py, _, _ in self.calibration_rows:
                        point = (round(px), round(py))
                        cv2.circle(frame, point, 3, (30, 215, 255), -1)
                        cv2.putText(frame, name, (point[0] + 4, point[1] - 3),
                                    cv2.FONT_HERSHEY_SIMPLEX, .35, (30, 215, 255), 1)
            if role == 'top':
                frame = self._tray_candidate_preview(frame)
            self.views[role].show_frame(frame)

    def _tray_candidate_preview(self, frame):
        overlay = self._tray_location_overlay
        if overlay is None:
            return frame
        if (time.monotonic() - overlay['timestamp'] >= 2
                or overlay['epoch'] != self.machine.epoch or overlay['source_top'] != self.sources['top']):
            self._tray_location_overlay = None
            return frame
        preview = frame.copy()
        height, width = preview.shape[:2]
        ordered = sorted(overlay['candidates'], key=lambda item: (
            item['selection_key'] is None, item['selection_key'] or (), item['index']))
        for rank, item in enumerate(ordered, 1):
            left, top, right, bottom = (int(round(value)) for value in item['box_full'])
            left, right = max(0, min(width - 1, left)), max(0, min(width - 1, right))
            top, bottom = max(0, min(height - 1, top)), max(0, min(height - 1, bottom))
            color = ((0, 0, 255) if item['index'] == overlay['blocked_index'] else
                     (0, 255, 0) if item['index'] == overlay['selected_index'] else (0, 255, 255))
            cv2.rectangle(preview, (left, top), (right, bottom), color, 2)
            cv2.putText(preview, '#%d' % rank, (left, max(12, top - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
        return preview

    def _show_competition_frames(self):
        if self._frames_at is None:
            self.release_competition.set_frames(None, None)
        else:
            self._competition_preview_at = self._frames_at
            self.release_competition.set_frames(self.frames['right'], self._tray_candidate_preview(self.frames['top']))

    def _vision_finished(self):
        if self.vision is not None and not self.vision.wait(0):
            QTimer.singleShot(20, self._vision_finished)
            return
        worker, self.vision = self.vision, None
        self._tray_background_capture = self._tray_location_hold = None
        self._tray_location_overlay = None
        if worker is not None:
            worker.deleteLater()
        self._job = None
        self._frames_at = None
        self._tray_status = '等待托盘定位'
        if self.release_competition is not None:
            self._show_competition_frames()
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
        if exec_qt(dialog) == QDIALOG_ACCEPTED:
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
        if not self._start_ready():
            return False
        if not self.machine.snapshot().origin_confirmed and not self._ask_origin_unchanged():
            return False
        if not self._start_ready():
            return False
        if not self.machine.snapshot().origin_confirmed:
            return self._begin_known_reset('start', clear_counts=False)
        return self._command(self.machine.start, '启动请求已发送。')

    def _ask_origin_unchanged(self, full_reset=False):
        yes = getattr(QMessageBox, 'Yes', None)
        no = getattr(QMessageBox, 'No', None)
        cancel = getattr(QMessageBox, 'Cancel', None)
        if yes is None:
            yes, no, cancel = (QMessageBox.StandardButton.Yes,
                               QMessageBox.StandardButton.No, QMessageBox.StandardButton.Cancel)
        message = ('固定零点和机构位置是否仍与上次记录一致？\n'
                   '“是”：零点未变，机构未被手推、没有卡住或丢步，按现有坐标继续。\n'
                   '“否”：零点或位置已改变，取消本次操作，重新确认实际机械零点。\n'
                   '“取消”：保持当前状态。')
        if full_reset:
            message += ('\n\n本次将停止流程、抬起推板、回 HOME 并复位，完成后清空箱号、计数和计时。'
                        '\n请先取走六箱中的物块，并移开工作范围内的手和工具。')
        elif not self.machine.snapshot().origin_confirmed and self.machine.can_reuse_origin:
            message += '\n\n设备已停止，本次先按保留的坐标回 HOME 并恢复零点，再启动。'
        answer = QMessageBox.question(self, '完全复位确认' if full_reset else '启动前零点确认',
                                      message, yes | no | cancel, cancel)
        if answer != yes:
            if answer == no:
                self._reset_action = None
                self._job = None
                self.machine.invalidate_origin()
            self.write_log('零点或机构位置已改变，请重新确认实际机械零点。' if answer == no else '操作已取消。')
            return False
        return True

    def _start_ready(self):
        reason = self._start_block_reason(self.machine.snapshot())
        if reason is None:
            return True
        self.write_log(reason)
        return False

    def _start_block_reason(self, snapshot):
        if self._closing:
            return '正在关闭操作台'
        if self._models_open() or self._disconnect_deadline is not None:
            return '模型管理或断开流程进行中'
        if self.vision is None:
            return '等待连接三路相机'
        if self.linux_20261107 and not self._models_warm:
            return '模型预热中，请等待'
        if self._frames_at is None or time.monotonic() - self._frames_at > 2:
            return '等待三路相机画面'
        if self._job is not None:
            return '视觉任务进行中'
        if not self.calibration_valid or not self.calibration_confirmed.isChecked():
            return '九点标定未通过校验，请导入或保存有效的九点标定。'
        if not self._round_available():
            return self._binding_error or '上一件投箱尚未确认，请核查六箱后清空本轮再开始。'
        if self.linux_20261107:
            origin_available = ((snapshot.origin_confirmed and snapshot.state == 'armed')
                                or (self.machine.can_reuse_origin
                                    and snapshot.state in ('stopped', 'needs_origin')))
            if (not self.machine.ready or not origin_available
                    or self.machine.reset_in_progress or snapshot.pusher_down or self.machine.pending):
                return '启动需要有效的坐标记录。若控制板重启、手推或卡步，请先恢复实际机械零点。'
            if self.auto_bins.total >= release.ROUND_SIZE:
                return '本轮已完成，请清空六箱并开启新一轮。'
        return None

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

    def toggle_feed_pause(self):
        paused = not self.machine.feed_pause_requested
        if not paused and not self._automatic_ready():
            self.write_log('继续需要相机、串口、标定和本轮记录均就绪。')
            return False
        return self._command(lambda: self.machine.set_feed_paused(paused),
                             '本件完成后暂停进料。' if paused else '继续自动分拣，等待 ESP32 请求推料。')

    def return_home(self):
        if self._closing or self._models_open() or self._disconnect_deadline is not None:
            self.write_log('当前正在关闭或更新模型，无法移动。')
            return False
        return self._command(lambda: self.machine.goto(0, 0), '已请求移到已确认的 HOME (0, 0)。')

    def stop(self):
        self._reset_deadline = None
        self._reset_action = None
        self._last_sort = None
        return self._command(self.machine.stop, '停止请求已发送；再次启动时确认零点和机构位置是否改变。')

    def reset(self):
        if not self.machine.ready or self._models_open() or self._closing or self._disconnect_deadline is not None:
            self.write_log('复位需要通信就绪且设备连接有效。')
            return False
        if self.machine.reset_in_progress or self.machine.pending:
            self.write_log('请等待当前动作结束或先停止，再完全复位。')
            return False
        if not self._ask_origin_unchanged(full_reset=True):
            return False
        return self._begin_known_reset('full', clear_counts=True)

    def _begin_known_reset(self, action, clear_counts):
        generation = self.machine.reset_generation
        if not self._command(lambda: self.machine.reset_known_origin(clear_counts=clear_counts),
                             '完全复位进行中：等待停机、抬起、HOME 到位和复位回执。'):
            return False
        self._reset_action = action
        self._reset_generation_seen = generation
        self._job = None
        self._scan_gate = None
        self._last_sort = None
        return True

    def _finish_reset_action(self):
        action = self._reset_action
        if action is None:
            return
        if self.machine.reset_generation == self._reset_generation_seen:
            if not self.machine.reset_in_progress:
                self._reset_action = None
                self.write_log('复位未完成，箱号和计数已保留。')
            return
        self._reset_action = None
        self._reset_generation_seen = self.machine.reset_generation
        if action == 'full':
            if not self.clear_round(confirmed=True):
                self.stop()
                return
            reset_clock = getattr(self, '_reset_debug_clock', None)
            if reset_clock is not None:
                reset_clock()
            self.write_log('完全复位完成：HOME 已确认，箱号、计数和计时已清零，等待启动。')
        elif action == 'start':
            if self._start_ready():
                self._command(self.machine.start, '已按保留坐标回 HOME 并确认零点，启动请求已发送。')

    def _request(self, kind):
        if (self.vision is None or self._frames_at is None or self._job is not None
                or self.machine.reset_in_progress):
            self.write_log('相机未就绪或已有视觉任务。')
            return False
        self._sequence += 1
        token = self._sequence
        self._job = (token, self.machine.epoch, kind)
        try:
            if kind == 'tray_background':
                self._tray_background_capture = (token, self.machine.epoch, self.sources['top'],
                                                 tuple(release.TRAY_FOREGROUND_ROI), self.vision)
                accepted = self.vision.capture_tray_background(token)
            elif kind in ('locate', 'manual_locate'):
                self._tray_location_hold = None
                if not self.calibration_valid:
                    raise ValueError('请先保存九点标定。')
                reference = self.tray_background_path if self.linux_20261107 and self.tray_background_path.is_file() else None
                if self.linux_20261107:
                    roi = release.TRAY_FOREGROUND_ROI if reference is not None else release.TRAY_ROI
                    offset = release.PUSH_OFFSET
                    bounds = MACHINE_REGION if reference is not None else release.PICK_REGION
                else:
                    roi, offset, bounds = TRAY_ROI, (4, 0), MACHINE_REGION
                calibration = validate_points(self.calibration_rows, roi=roi, physical_region=bounds)
                if self.linux_20261107:
                    x, y, width, height = self.calibration_frame['box']
                    point_x = {row[0]: float(row[3]) for row in self.calibration_rows}
                    column_x = [float(np.median([point_x['P%d' % point] for point in column]))
                                for column in ((1, 4, 7), (2, 5, 8), (3, 6, 9))]
                    if (not np.isfinite(column_x).all()
                            or not column_x[0] < column_x[1] < column_x[2]):
                        raise ValueError('九点标定的三列机械 X 必须有限且从 P1 列到 P3 列递增')
                    column_boundaries = tuple((left + right) / 2
                                              for left, right in zip(column_x, column_x[1:]))
                    policy = dict(confidence=release.TRAY_CONFIDENCE,
                                  selection_region=(x, y, x + width, y + height),
                                  column_boundaries=column_boundaries)
                    if reference is not None:
                        policy.update(background_reference=str(reference), selection_region=None, motion_target_only=True)
                    accepted = self.vision.locate(token, calibration, roi, offset, **policy)
                else:
                    accepted = self.vision.locate(token, calibration, roi, offset)
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

    def save_tray_background(self):
        state = self.machine.snapshot().state
        if (self.vision is not None
                and getattr(self.vision, 'sources', self.sources).get('top') != self.sources['top']):
            self.write_log('托盘相机设置已更改，请重新连接相机后再保存底板。')
            return False
        if (not self.linux_20261107 or state not in ('disconnected', 'needs_origin', 'armed', 'stopped')
                or self.machine.pending or self.machine.reset_in_progress or self._closing
                or self._models_open() or self._disconnect_deadline is not None
                or self.vision is None or self._frames_at is None
                or time.monotonic() - self._frames_at > 2
                or (self.machine.connected and not self.machine.idle_confirmed)):
            self.write_log('保存托盘底板需要停机、无待完成动作且相机画面就绪；请先清空托盘。')
            return False
        self.write_log('采集已清空托盘的新鲜底板画面；仅保存参考，不发送运动命令。')
        return self._request('tray_background')

    def on_tray_background_ready(self, token, frame):
        capture = self._tray_background_capture
        kind = self._take_job(token)
        if capture is not None and capture[0] == token:
            self._tray_background_capture = None
        if (kind != 'tray_background' or capture is None or capture[0] != token
                or capture[2] != self.sources['top'] or capture[4] is not self.vision
                or self._closing or self._disconnect_deadline is not None
                or self.machine.pending or self.machine.reset_in_progress
                or (self.machine.connected and not self.machine.idle_confirmed)
                or self.machine.snapshot().state not in ('disconnected', 'needs_origin', 'armed', 'stopped')):
            self._persist_log('丢弃过期托盘底板结果 token=%s' % token)
            return
        try:
            from tray_background import TrayBackgroundReference
            TrayBackgroundReference(frame, capture[2], capture[3]).save(self.tray_background_path)
            self.write_log('托盘底板已保存：' + str(self.tray_background_path))
        except (OSError, ValueError, TypeError) as error:
            self.write_log('托盘底板保存失败：' + str(error))

    def on_location_evidence(self, token, evidence):
        job = self._job
        if (job is None or job[0] != token or job[1] != self.machine.epoch
                or job[2] not in ('locate', 'manual_locate')):
            return
        locator = evidence.get('locator') or {}
        self._tray_location_overlay = None
        if evidence.get('source_top') == self.sources['top']:
            candidates = []
            for record in locator.get('candidates', []):
                box = record.get('box_full', record.get('bbox_full'))
                if box is not None and len(box) == 4 and np.isfinite(box).all():
                    key = record.get('selection_key')
                    candidates.append({'index': record['index'], 'box_full': tuple(box),
                                       'selection_key': tuple(key) if key is not None else None})
            self._tray_location_overlay = {'epoch': job[1], 'source_top': evidence['source_top'],
                'timestamp': time.monotonic(), 'candidates': candidates,
                'selected_index': locator.get('selected_index'), 'blocked_index': locator.get('blocked_index')}
        self._tray_location_hold = (token, job[1], locator.get('hold_reason'))
        detector = dict(evidence.get('detector') or {})
        mask = detector.pop('foreground_mask', None)
        tray = {key: value for key, value in evidence.items() if key not in ('captures', 'detector')}
        tray['detector'] = detector
        result = SimpleNamespace(diagnostics={'captures': evidence.get('captures') or {},
                                             'tray': tray, 'foreground_mask': mask})
        try:
            path = self.diagnostics.save_sample(result, {'token': token, 'epoch': job[1], 'kind': job[2]}, group='tray')
            self._persist_log('托盘定位诊断已保存：' + str(path))
        except (OSError, ValueError, TypeError, cv2.error) as error:
            self.write_log('托盘定位诊断保存失败：' + str(error))

    def _dispatch_vision(self):
        if self._job is not None:
            return
        state = self.machine.snapshot().state
        if state in ('locating', 'recognizing'):
            if state == 'recognizing':
                if self._scan_epoch != self.machine.epoch:
                    self._scan_epoch = self.machine.epoch
                    self._scan_gate = ScanConsensus(max_attempts=release.SCAN_ATTEMPTS if self.linux_20261107 else 8)
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
        evidence = getattr(result, 'diagnostics', None) or {}
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
            self._shape_name(shape), color_name, confidence, result.elapsed_ms, surface_summary(result)))
        scene_text = '已检出物块' if detection is not None else '未识别到有效目标，等待模型复核'
        self.result_label.setText(scene_text + '\n' + self.result_label.text())
        self.write_log(scene_text)
        sources = {'mask': '表面掩膜', 'box': '检测框（覆膜未参与）', 'none': '未取色'}
        color_sources = getattr(result, 'color_sources', ('none', 'none'))
        self.write_log(surface_summary(result) + '；取色 左：%s / 右：%s' % tuple(
            sources.get(source, source) for source in color_sources))
        self.write_log('识别结果 token=%s kind=%s shape=%s shape_conf=%s color=%s color_conf=%s pixels=%s elapsed_ms=%.1f' % (
            token, kind, shape, getattr(detection, 'confidence', None), color_name,
            getattr(color, 'confidence', None), getattr(color, 'valid_pixels', None), result.elapsed_ms))
        self.preview.show_frame(result.preview_bgr)
        if self.release_competition is not None:
            self.debug_item_label.setText('货物名称：%s %s' % (color_name, self._shape_name(shape)))
            sample = evidence.get('color_sample_bgr')
            if sample is None:
                self.color_view.frame = None
                self.color_view.refresh()
            else:
                self.color_view.show_frame(sample)
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
                if self.linux_20261107:
                    if self.machine.reject_unknown_scan():
                        self.write_log('%d 次复核未识别为稳定类别，正在打开底板排出未知料；'
                                       '等待关板确认后继续推下一件，不分配箱号、不计成功数量。' % self._scan_gate.attempts)
                        self.update_state()
                    else:
                        self.stop()
                        self.write_log('未知料排出未执行：' + (self.machine.snapshot().last_error or '设备状态不允许'))
                else:
                    self.stop()
                    self.write_log('%d 次复核未得到连续一致的有效形状和颜色，未分配箱号、未投箱。请导出诊断。' % self._scan_gate.attempts)
                return
            if self.linux_20261107:
                self.machine.tick()
                if self.machine.snapshot().state != 'recognizing':
                    self.write_log(self.machine.snapshot().last_error or '扫描已结束，未分配箱号。')
                    return
            new_binding = (shape, color_name) not in self.bindings
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
                if self.release_competition is not None:
                    self._set_current_item(shape, color_name, bin_id, self.auto_bins.total + 1)
                    self._refresh_binding_display()
                self.write_log('%s %s → %s号箱，等待投箱完成。' % (color_name, shape, bin_id))
            else:
                # A deadline rejection stops locally before writing SORT.
                # A serial write error disconnects, so its pending record stays.
                if (self.linux_20261107 and not self.machine.ready
                        and self.machine.snapshot().state == 'stopped'):
                    try:
                        self.auto_bins.cancel_unsent(bin_id, new_binding=new_binding)
                        self._refresh_binding_display()
                    except (OSError, ValueError) as error:
                        self.write_log('未发送投箱，但撤回预留记录失败：' + str(error))
                self.stop()

    def on_located(self, token, target):
        hold = self._tray_location_hold
        hold_reason = hold[2] if hold is not None and hold[:2] == (token, self.machine.epoch) else None
        kind = self._take_job(token)
        if kind is None:
            return
        self._tray_location_hold = None
        self._tray_status = (str(hold_reason) if target is None and hold_reason
                             else '无目标' if target is None else '目标 %.1f, %.1f mm' % target)
        self.write_log('托盘' + self._tray_status, view=self.tray_log_view)
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
        if self.release_competition is not None:
            captures = (getattr(result, 'diagnostics', None) or {}).get('captures') or {}
            frame = captures.get('right')
            self._round_images[bin_id] = (result.preview_bgr if frame is None else frame).copy()
            self._set_current_item(shape, color, bin_id, total)
        else:
            self._append_result_row(shape, color, bin_id, total, result.preview_bgr)
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
            return
        if self.linux_20261107 and total >= release.ROUND_SIZE:
            self.stop()
            self.write_log('本轮 %d 件已确认投放，已结束。' % total)

    def _append_result_row(self, shape, color, bin_id, total, frame):
        row = self.records.rowCount()
        self.records.insertRow(row)
        for column, value in ((0, total), (1, '%s %s → %s号盒' % (color, self._shape_name(shape), bin_id)), (3, total)):
            self.records.setItem(row, column, QTableWidgetItem(str(value)))
        image = ImageView('')
        image.setFixedHeight(64)
        image.show_frame(frame)
        self.records.setCellWidget(row, 2, image)
        self.records.setRowHeight(row, 68)
        if self.records.rowCount() > 200:
            self.records.removeRow(0)

    def _set_current_item(self, shape, color, bin_id, sequence):
        name = self._shape_name(shape)
        self._current_item = {'sequence': sequence, 'name': '%s %s' % (color, name),
                              'bin': '%s号' % bin_id, 'count': self.auto_bins.counts[bin_id - 1],
                              'color': color, 'shape': name}
        self.debug_sequence_label.setText('序号：%s' % sequence)
        self.debug_item_label.setText('货物名称：' + self._current_item['name'])

    def _shape_name(self, shape):
        return self._shape_profile.SHAPES.get(shape, shape) if self._shape_profile is not None else shape

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
                    and self.machine.snapshot().state in ('wait_done',)
                    and (pending['shape'], pending['color'], pending['bin']) == self._last_sort[:3])

    def _refresh_binding_display(self):
        if self.release_competition is not None:
            self._refresh_release_bins()
            return
        if self.auto_bins is None:
            self.total_label.setText('本轮记录待检查')
            for number, label in enumerate(self.bin_labels, 1):
                label.setText('%s号箱 · 待检查' % number)
                label.setToolTip(self._binding_error or '尚未读取本轮记录')
            return
        entries = {row['bin']: row for row in self.auto_bins.entries} if self.auto_bins else {}
        pending = self.auto_bins.pending if self.auto_bins else None
        total = self.auto_bins.total if self.auto_bins else 0
        self.total_label.setText('本轮已投放 %s/%s' % (total, release.ROUND_SIZE) if self.linux_20261107 else '本轮成功总数 %s' % total)
        for number, label in enumerate(self.bin_labels, 1):
            row = entries.get(number)
            label.setText('%s号箱 · %s件' % (number, row['count'] if row else 0))
            label.setToolTip('%s %s' % (row['color'], row['shape']) if row else '按首次出现顺序自动绑定')
            if row:
                label.setText(label.text() + '\n%s %s' % (row['color'], self._shape_name(row['shape'])))
            if pending is not None and pending['bin'] == number:
                label.setText(label.text() + ' · 待确认')

    def _refresh_release_bins(self):
        entries = {row['bin']: row for row in self.auto_bins.entries} if self.auto_bins else {}
        pending = self.auto_bins.pending if self.auto_bins else None
        total = self.auto_bins.total if self.auto_bins else 0
        bins = []
        for number, label in enumerate(self.debug_bin_labels, 1):
            row = entries.get(number)
            name = '%s %s' % (row['color'], self._shape_name(row['shape'])) if row else '待绑定'
            if self.auto_bins is None:
                name = '待检查'
            if pending is not None and pending['bin'] == number:
                name += ' · 待确认'
            count = row['count'] if row else 0
            label.setText('%s号盒  %s  数量：%s' % (number, name, count))
            bins.append((name, count, self._round_images.get(number)))
        self.debug_total_label.setText('分拣成功总数量：%s' % total)
        self._release_status = self._release_status_values(self.machine.snapshot())
        status, serial_status, tray_status = self._release_status
        current = dict(self._current_item, serial=serial_status, tray=tray_status)
        self.release_competition.render(total, bins, status, current)

    def clear_round(self, *, confirmed=False):
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
        if not confirmed:
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
        if self.release_competition is not None:
            self._round_images.clear()
            self._current_item.clear()
            self.debug_sequence_label.setText('序号：—')
            self.debug_item_label.setText('货物名称：—')
        else:
            self.records.setRowCount(0)
        self._refresh_binding_display()
        self.write_log('已开始新一轮，按形状＋颜色首次出现顺序自动分配1至6号箱。')
        return True

    def binding_dialog(self):
        dialog = self.create_binding_dialog()
        exec_qt(dialog)
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
        exec_qt(dialog)
        dialog.deleteLater()

    def save_calibration(self, rows, frame=None):
        validate_points(rows)
        frame = normalize_frame_settings(self.calibration_frame if frame is None else frame)
        atomic_json(self.calibration_path, {'mode': 'live', 'points': rows, 'frame': frame})
        self.calibration_rows = rows
        self.calibration_frame = frame
        self.calibration_valid = True
        self.calibration_confirmed.setChecked(True)
        self.write_log('九点标定已保存并通过校验，无需手动勾选。')

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
                self.calibration_confirmed.setChecked(True)
                self.write_log('已自动加载保存的九点标定，数据校验通过。')
        except (KeyError, TypeError, ValueError, OSError) as error:
            self.write_log('配置未加载：' + str(error))

    def open_models(self):
        if self._models_open():
            if self._model_restart_required:
                self.write_log('模型配置已改变：请用“更多→退出”关闭上位机后重新打开；本窗口不再连接设备。')
            elif self._model_process is not None:
                self.write_log('模型管理窗口已经打开，但可能被全屏界面挡住：请按 Alt+Tab 切换，'
                               '或先点“取消全屏”。')
            else:
                self.write_log('正在关闭相机与串口，随后自动打开模型管理，请稍候。')
            return
        self._model_pending = True
        self.close_serial()
        if self.vision is not None:
            self.vision.shutdown()
        elif self.serial_port is None and self._model_pending:
            self._launch_models()

    def _models_open(self):
        if self._model_process is not None and self._model_process.poll() is not None:
            self._model_process = None
            self._restore_fullscreen_after_models()
            from side_model_store import SideModelStore
            try:
                active = SideModelStore(ROOT).read_active()
                old_version = (self._active_side_model or {}).get('version')
                if (active or {}).get('version') != old_version:
                    self._model_restart_required = True
                    self.write_log('模型配置已改变，必须退出并重启完整上位机；当前窗口禁止重新连接设备。')
            except Exception as exc:
                self._model_restart_required = True
                self.write_log('模型配置校验失败，禁止重新连接设备：' + str(exc))
        return self._model_restart_required or self._model_pending or self._model_process is not None

    def _launch_models(self):
        self._model_pending = False
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            path = self.data_dir / 'model_trial_raw.png'
            from legacy_vision import prepare_legacy_pair
            raw, _, _ = prepare_legacy_pair(self.frames['left'], self.frames['right'])
            if not cv2.imwrite(str(path), raw):
                raise OSError('无法保存模型试跑图片')
            arguments = [sys.executable, str(ROOT / 'side_model_manager.py')]
            # A full-screen console can end up covering the child window on
            # this desktop, leaving the operator with an unreachable manager
            # that still counts as open. Leave full screen first, then restore
            # it once the manager process has exited.
            if self.isFullScreen():
                self._restore_fullscreen = True
                self.showNormal()
                self.raise_()
                self.activateWindow()
            self._model_process = subprocess.Popen(arguments, cwd=str(ROOT))
            self.write_log('已打开正式上位机模型管理（窗口可能在新窗口，按 Alt+Tab 切换）。'
                           '确认启用后需退出并重新打开上位机；相机、串口不会自动重连。')
        except (OSError, ValueError) as error:
            self.write_log('模型窗口启动失败：' + str(error))

    def _restore_fullscreen_after_models(self):
        if self._restore_fullscreen:
            self._restore_fullscreen = False
            self.showFullScreen()
            self.raise_()
            self.activateWindow()

    def show_competition(self):
        self.pages.setCurrentWidget(self.competition_page)
        if self.release_competition is not None:
            self._show_competition_frames()

    def show_debug(self):
        self.pages.setCurrentWidget(self.debug_page)
        if self.release_competition is not None and self._frames_at is not None:
            self._show_debug_frames()

    def closeEvent(self, event):
        if not self._closing:
            self._closing = True
            self._startup_timer.stop()
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
    parser.add_argument('--startup-serial-port', metavar='DEVICE', help='启动时打开指定串口和已配置的三路相机')
    parser.add_argument('--fixed-origin-on-boot', action='store_true', help='开机从固定机械零点上电时自动确认零点')
    parser.add_argument('--new-round-on-boot', action='store_true', help='设备重新开机后清空六箱计数和绑定')
    models = parser.add_mutually_exclusive_group()
    models.add_argument('--linux-20261107', action='store_true', help='使用 Linux 8 类模型和比赛流程')
    models.add_argument('--butu-24', action='store_true', help='使用补图训练的 24 类形状模型')
    args = parser.parse_args()
    if args.fixed_origin_on_boot and not args.startup_serial_port:
        parser.error('--fixed-origin-on-boot 需要 --startup-serial-port')
    try:
        round_boot_id = _system_boot_id() if args.new_round_on_boot else None
    except OSError as error:
        parser.error('无法读取 Jetson 本次开机标识：' + str(error))
    application = QApplication(sys.argv[:1])
    window = LiveWindow(linux_20261107=args.linux_20261107, butu_24=args.butu_24,
                        startup_serial_port=args.startup_serial_port,
                        fixed_origin_on_boot=args.fixed_origin_on_boot and _claim_fixed_origin_boot(),
                        round_boot_id=round_boot_id)
    window.showFullScreen() if args.fullscreen else window.show()
    return exec_qt(application)


if __name__ == '__main__':
    raise SystemExit(main())
