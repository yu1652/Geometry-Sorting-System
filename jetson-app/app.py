"""1024x600 Qt operator application, phase 1: offline replay only."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

try:
    from PySide6.QtCore import Qt, QTimer, Slot
    from PySide6.QtGui import QImage, QPixmap, QKeySequence, QShortcut
    from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QPushButton,
        QVBoxLayout, QHBoxLayout, QFrame, QFileDialog, QMessageBox, QDialog, QTabWidget,
        QTextEdit, QComboBox, QFormLayout, QDialogButtonBox)
    QT_ALIGN_CENTER = Qt.AlignmentFlag.AlignCenter
    QT_KEEP_ASPECT = Qt.AspectRatioMode.KeepAspectRatio
    QT_SMOOTH = Qt.TransformationMode.SmoothTransformation
    QIMAGE_RGB888 = QImage.Format.Format_RGB888
    QDIALOG_ACCEPTED = QDialog.DialogCode.Accepted
    QBUTTON_OK = QDialogButtonBox.StandardButton.Ok
    QBUTTON_CANCEL = QDialogButtonBox.StandardButton.Cancel
except ImportError:
    from PyQt5.QtCore import Qt, QTimer, pyqtSlot as Slot
    from PyQt5.QtGui import QImage, QPixmap, QKeySequence
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QPushButton,
        QVBoxLayout, QHBoxLayout, QFrame, QFileDialog, QMessageBox, QDialog, QTabWidget,
        QTextEdit, QComboBox, QFormLayout, QDialogButtonBox, QShortcut)
    QT_ALIGN_CENTER = Qt.AlignCenter
    QT_KEEP_ASPECT = Qt.KeepAspectRatio
    QT_SMOOTH = Qt.SmoothTransformation
    QIMAGE_RGB888 = QImage.Format_RGB888
    QDIALOG_ACCEPTED = QDialog.Accepted
    QBUTTON_OK = QDialogButtonBox.Ok
    QBUTTON_CANCEL = QDialogButtonBox.Cancel

# On Jetson/Xavier, importing OpenCV before Qt can trigger a static TLS
# allocation failure in libGLdispatch.  Load the Qt bindings first.
import cv2

from core import ReplayState, save_sample
from media import MediaSource
from vision_worker import VisionWorker, default_recognizer_factory

ROOT = Path(__file__).resolve().parent
STYLE = '''
QMainWindow,QDialog {background:#eaf0f5;color:#233246;}
QWidget {font-family:"Microsoft YaHei","Noto Sans CJK SC",sans-serif;font-size:12px;}
QFrame#panel {background:white;border:1px solid #d6e0e9;border-radius:7px;}
QLabel#brand {font-size:17px;font-weight:700;}
QLabel#result {font-size:29px;font-weight:700;color:#147d73;}
QLabel#muted {color:#667b8c;}
QLabel#image {background:#19343e;color:#c3d4dc;border-radius:4px;}
QPushButton {background:white;border:1px solid #cbd7e2;border-radius:5px;padding:7px 10px;min-height:19px;}
QPushButton:hover {background:#e2eeed;}
QPushButton#go {background:#147d73;color:white;border-color:#147d73;}
QPushButton#stop {background:#a74747;color:white;border-color:#a74747;}
QPushButton:disabled {color:#a4b0bb;background:#eef2f5;}
QTextEdit {border:0;background:white;color:#526a7d;font-size:12px;}
QComboBox {background:white;padding:5px;border:1px solid #ccd8e1;}
'''


class ImageView(QLabel):
    def __init__(self, placeholder):
        super().__init__(placeholder)
        self.placeholder = placeholder
        self.setObjectName('image')
        self.setAlignment(QT_ALIGN_CENTER)
        self.setMinimumSize(1, 1)
        self.frame = None

    def show_frame(self, frame):
        self.frame = frame.copy()
        self.refresh()

    def refresh(self):
        if self.frame is None:
            self.clear()
            self.setText(self.placeholder)
            return
        rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        image = QImage(rgb.data, w, h, rgb.strides[0], QIMAGE_RGB888).copy()
        self.setPixmap(QPixmap.fromImage(image).scaled(self.size(), QT_KEEP_ASPECT,
                                                      QT_SMOOTH))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh()


def panel(title, child):
    box = QFrame()
    box.setObjectName('panel')
    layout = QVBoxLayout(box)
    layout.setContentsMargins(8, 6, 8, 6)
    layout.setSpacing(5)
    heading = QLabel(title)
    heading.setObjectName('muted')
    layout.addWidget(heading)
    layout.addWidget(child, 1)
    return box


class ModelDialog(QDialog):
    """Offline model trial and explicit activation."""
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.closed = False
        self.busy = False
        self.pending = False
        self.mutating = False
        self.paths = {}
        self.setWindowTitle('模型管理')
        self.resize(700, 500)
        layout = QVBoxLayout(self)
        self.labels = {}
        try:
            active = window._thread._store.read_active()['active']
        except Exception:
            active = {}
        for slot, title in (('side_detector', '形状'), ('surface_mask', '掩膜')):
            row = QHBoxLayout()
            record = active.get(slot) or {}
            label = QLabel('%s：%s（未验证）' % (title, record.get('version', '默认/未配置')))
            self.labels[slot] = label
            row.addWidget(label, 1)
            choose = QPushButton('选择模型')
            choose.clicked.connect(lambda checked=False, s=slot: self.choose(s))
            row.addWidget(choose)
            trial = QPushButton('试跑')
            trial.clicked.connect(lambda checked=False, s=slot: self.start_trial(s))
            row.addWidget(trial)
            rollback = QPushButton('回退')
            rollback.clicked.connect(lambda checked=False, s=slot: self.rollback(s))
            row.addWidget(rollback)
            layout.addLayout(row)
        self.status = QLabel('请选择候选 manifest 和双视图样本')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.preview = ImageView('候选预览')
        layout.addWidget(self.preview, 1)
        row = QHBoxLayout()
        self.confirm_button = QPushButton('确认启用')
        self.confirm_button.setEnabled(False)
        self.confirm_button.clicked.connect(self.confirm)
        row.addWidget(self.confirm_button)
        cancel = QPushButton('取消候选')
        cancel.clicked.connect(self.cancel)
        row.addWidget(cancel)
        layout.addLayout(row)
        window._thread.model_trial_ready.connect(self.on_trial)
        window._thread.model_changed.connect(self.on_changed)
        window._thread.model_failed.connect(self.on_failed)
        window.model_dialog = self

    def fixture(self):
        w = self.window
        if w.frame is None or not w.layout_mode.startswith('双') or w.frame.shape[1] < 2:
            self.status.setText('请先加载双视图样本')
            return None
        w.pause()
        mid = w.frame.shape[1] // 2
        frames = (w.frame[:, :mid].copy(), w.frame[:, mid:].copy())
        if w.input_mode.currentIndex() == 1:
            return (*frames, True)
        return frames

    def available(self):
        return not self.closed and not self.busy and self.window.active_request_id is None and not self.window.closing

    def choose(self, slot):
        if not self.available() or self.pending: return
        path, _ = QFileDialog.getOpenFileName(self, '选择 manifest', '', 'JSON (*.json)')
        if path:
            self.paths[slot] = path
            self.status.setText(path)

    def start_trial(self, slot):
        if not self.available() or self.pending: return
        if slot not in self.paths:
            self.status.setText('请先选择候选 manifest')
            return
        fixture = self.fixture()
        if fixture is None: return
        self.busy = True
        self.window.model_busy = True
        self.status.setText('正在试跑…')
        self.window._thread.trial_candidate(self.paths[slot], slot, fixture)

    def on_trial(self, payload):
        if self.closed or not self.busy: return
        self.busy = False
        self.pending = True
        result = payload['result']
        self.status.setText('%s / %s · %s · %.1f ms；请检查后确认' % (payload['slot'], payload['version'], ', '.join(result.mask_status), result.elapsed_ms))
        self.preview.show_frame(result.preview_bgr)
        self.confirm_button.setEnabled(True)

    def confirm(self):
        if not self.pending or self.busy or self.closed: return
        self.pending = False
        self.busy = True
        self.mutating = True
        self.status.setText('正在启用模型，请等待结果…')
        self.confirm_button.setEnabled(False)
        self.window._thread.confirm_candidate()

    def cancel(self):
        if self.closed or self.busy: return
        self.window._thread.cancel_candidate()
        self.pending = False
        self.window.model_busy = False
        self.confirm_button.setEnabled(False)
        self.status.setText('候选已取消')

    def rollback(self, slot):
        if not self.available() or self.pending: return
        fixture = self.fixture()
        if fixture is None: return
        self.busy = True
        self.window.model_busy = True
        self.mutating = True
        self.status.setText('正在验证回退…')
        self.window._thread.rollback(slot, fixture)

    def on_changed(self, active):
        if self.closed: return
        self.busy = self.pending = self.mutating = False
        self.window.model_busy = False
        for slot, label in self.labels.items():
            label.setText('%s：%s' % (slot, (active.get(slot) or {}).get('version', '未配置')))
        self.window._invalidate_frame()
        self.status.setText('模型已更新')

    def on_failed(self, message):
        if self.closed: return
        self.busy = self.pending = self.mutating = False
        self.window.model_busy = False
        self.confirm_button.setEnabled(False)
        self.status.setText('模型操作失败：' + message)

    def closeEvent(self, event):
        if self.mutating:
            self.status.setText('模型操作进行中，请等待结果')
            event.ignore()
            return
        super().closeEvent(event)

    def done(self, result):
        if self.mutating:
            self.status.setText('模型操作进行中，请等待结果')
            return
        if not self.closed:
            self.closed = True
            self.confirm_button.setEnabled(False)
            self.window._thread.cancel_candidate()
            # A queued cancel follows any in-flight trial on the worker.
            self.window.model_busy = self.busy
            self.window.model_dialog = None
        super().done(result)


class Window(QMainWindow):

    def __init__(self, recognizer_factory=None):
        super().__init__()
        self.setWindowTitle('YLERS · Jetson Xavier NX 视觉操作台 — 离线版')
        self.resize(1024, 600)
        self.setMinimumSize(1024, 600)
        self.source = MediaSource()
        self.state = ReplayState()
        self.frame = None
        self.frame_revision = 0
        self.request_sequence = 0
        self.active_request_id = None
        self.closing = False
        self.model_busy = False
        self.model_dialog = None
        self._close_ready = False
        self._thread = VisionWorker(recognizer_factory or default_recognizer_factory, self)
        self._thread.completed.connect(self._on_recognized)
        self._thread.failed.connect(self._on_recognition_failed)
        self._thread.finished.connect(self._worker_thread_finished)
        self._thread.model_trial_ready.connect(self._discard_closed_model_operation)
        self._thread.model_changed.connect(self._discard_closed_model_operation)
        self._thread.model_failed.connect(self._discard_closed_model_operation)
        self._thread.start()
        self.last_saved = None
        self.layout_mode = '双视图拼接（中线拆分）'
        self.logs = []
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        self.setStyleSheet(STYLE)
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(10, 8, 10, 8)
        outer.setSpacing(8)
        top = QHBoxLayout()
        brand = QLabel('YLERS / 智能分拣')
        brand.setObjectName('brand')
        top.addWidget(brand)
        self.mode_label = QLabel('离线回放 · 机械输出关闭')
        self.mode_label.setStyleSheet('color:#29786d;background:#deefeb;padding:5px 10px;border-radius:4px')
        top.addWidget(self.mode_label)
        self.input_mode = QComboBox()
        self.input_mode.addItems(['原始彩色双摄图', '已处理特征拼图'])
        self.input_mode.setToolTip('原图自动生成灰度、增强和轮廓通道；特征拼图直接送入形状模型')
        top.addWidget(self.input_mode)
        top.addStretch()
        top.addWidget(QLabel('设备：未连接　累计：—'))
        outer.addLayout(top)
        body = QHBoxLayout()
        body.setSpacing(8)
        left = QVBoxLayout()
        left.setSpacing(8)
        images = QHBoxLayout()
        images.setSpacing(8)
        self.left_view = ImageView('请打开双视图图片或录像')
        self.right_view = ImageView('右路画面未加载')
        images.addWidget(panel('左轴测 / 原图 · 覆膜未就绪', self.left_view))
        images.addWidget(panel('右轴测 / 原图 · 覆膜未就绪', self.right_view))
        left.addLayout(images, 1)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.document().setMaximumBlockCount(200)
        logs = panel('运行记录', self.log_view)
        logs.setFixedHeight(110)
        left.addWidget(logs)
        body.addLayout(left, 2)
        side = QVBoxLayout()
        side.setSpacing(8)
        info = QWidget()
        fields = QVBoxLayout(info)
        fields.setContentsMargins(4, 0, 4, 0)
        self.result_label = QLabel('等待识别')
        self.result_label.setObjectName('result')
        fields.addWidget(self.result_label)
        self.color_label = QLabel('颜色：—　置信度：—')
        fields.addWidget(self.color_label)
        self.mask_status_label = QLabel('左覆膜：未就绪　右覆膜：未就绪')
        self.mask_status_label.setWordWrap(True)
        fields.addWidget(self.mask_status_label)
        self.preview_view = ImageView('识别预览将在此显示')
        fields.addWidget(self.preview_view)
        self.status_label = QLabel('请先打开图片或录像')
        self.status_label.setObjectName('muted')
        fields.addWidget(self.status_label)
        result_panel = panel('最终结果 / 设备状态', info)
        result_panel.setFixedHeight(174)
        side.addWidget(result_panel)
        self.tray_view = ImageView('俯拍摄像头未连接\nXY坐标：—')
        side.addWidget(panel('俯拍定位', self.tray_view), 1)
        body.addLayout(side, 1)
        outer.addLayout(body, 1)
        bins = QHBoxLayout()
        bins.setSpacing(8)
        for i in range(1, 7):
            label = QLabel(f'{i}号盒\n未绑定　数量：—')
            label.setStyleSheet('background:white;border:1px solid #d6e0e9;border-radius:5px;padding:7px')
            bins.addWidget(label, 1)
        outer.addLayout(bins)
        controls = QHBoxLayout()
        controls.setSpacing(6)
        self.buttons = {}
        for text, action, name in [
            ('打开文件', self.open_dialog, ''), ('开始回放', self.play, 'go'),
            ('暂停', self.pause, ''), ('停止回放', self.stop, 'stop'),
            ('下一帧', self.step, ''), ('手动识别', self.recognize, ''),
            ('保存样本', self.save, ''), ('设置 / 日志', self.settings, '')]:
            button = QPushButton(text)
            button.setObjectName(name)
            button.clicked.connect(action)
            controls.addWidget(button)
            self.buttons[text] = button
        outer.addLayout(controls)
        footer = QLabel('离线识别验证　｜　F5回放 / 空格暂停 / Esc停止　｜　自动分拣待实机接入')
        footer.setObjectName('muted')
        outer.addWidget(footer)
        self.shortcuts = []
        for key, action in [('F5', self.play), ('Space', self.pause), ('Escape', self.stop), ('Ctrl+O', self.open_dialog)]:
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(action)
            self.shortcuts.append(shortcut)
        self.log('程序已启动。离线模式不发送任何机械指令。')
        self.input_mode.currentIndexChanged.connect(self._input_mode_changed)

    def _input_mode_changed(self, index):
        self.pause()
        self._invalidate_frame()
        self.log('输入类型：' + self.input_mode.currentText())

    def log(self, message):
        line = datetime.now().strftime('%H:%M:%S') + '　' + message
        self.logs.append(line)
        self.logs = self.logs[-200:]
        self.log_view.setPlainText('\n'.join(self.logs[-30:]))
        self.log_view.verticalScrollBar().setValue(self.log_view.verticalScrollBar().maximum())

    def error(self, message):
        self.log(message)
        QMessageBox.warning(self, '操作提示', message)

    def open_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, '打开图片或录像', '',
            '图片与录像 (*.jpg *.jpeg *.png *.bmp *.webp *.mp4 *.avi *.mkv *.mov);;所有文件 (*)')
        if path:
            self.load_file(path)

    def load_file(self, path):
        self._invalidate_frame()
        self.timer.stop()
        try:
            self.source.open(path)
            frame = self.source.read()
            if frame is None:
                raise ValueError('文件未包含可读取的画面')
            self.state.load(str(path))
            self.display(frame)
            self.status_label.setText('已加载 / 暂停中')
            self.log('已加载：' + Path(path).name)
            return True
        except Exception as exc:
            self.source.close()
            self.state = ReplayState()
            self.frame = None
            for view in (self.left_view, self.right_view):
                view.frame = None
                view.refresh()
            self.status_label.setText('加载失败')
            self.error(str(exc))
            return False

    def display(self, frame):
        self._invalidate_frame()
        self.frame = frame.copy()
        if self.layout_mode.startswith('双'):
            midpoint = frame.shape[1]//2
            self.left_view.show_frame(frame[:, :midpoint])
            self.right_view.show_frame(frame[:, midpoint:])
        else:
            self.left_view.show_frame(frame)
            self.right_view.frame = None
            self.right_view.refresh()

    def play(self):
        try:
            self.state.play()
        except ValueError as exc:
            self.error(str(exc))
            return
        if not self.source.is_video:
            self.state.pause()
            self.log('当前是静态图片；可保存样本或打开录像。')
            return
        if self.source.capture.get(cv2.CAP_PROP_POS_FRAMES) >= self.source.capture.get(cv2.CAP_PROP_FRAME_COUNT):
            path = self.state.source
            if not self.load_file(path):
                return
            self.state.play()
        self.timer.start(max(1, round(1000/self.source.fps)))
        self.status_label.setText('录像回放中')

    def pause(self):
        self.timer.stop()
        self.state.pause()
        self.status_label.setText('已暂停' if self.state.source else '请先打开图片或录像')

    def stop(self):
        self.timer.stop()
        self.state.stop()
        self.status_label.setText('回放已停止' if self.state.source else '请先打开图片或录像')
        self.log('停止离线回放。')

    def step(self):
        self.pause()
        if self.source.is_video:
            self.next_frame()
        else:
            self.log('下一帧仅用于录像。')

    def next_frame(self):
        frame = self.source.read()
        if frame is None:
            self.pause()
            self.status_label.setText('录像已结束')
            return
        self.display(frame)

    def recognize(self):
        if self.frame is None:
            self.status_label.setText('请先加载双摄拼图')
            return
        if not self.layout_mode.startswith('双'):
            self.status_label.setText('手动识别需要双摄拼图布局')
            return
        if self.active_request_id is not None or self.closing or self.model_busy:
            return
        self.pause()
        midpoint = self.frame.shape[1] // 2
        if midpoint == 0 or midpoint == self.frame.shape[1]:
            self.status_label.setText('手动识别需要双摄拼图布局')
            return
        self.request_sequence += 1
        self.active_request_id = self.request_sequence
        self.status_label.setText('正在识别…')
        self.buttons['手动识别'].setEnabled(False)
        self._thread.recognize(self.active_request_id, self.frame_revision,
                               self.frame[:, :midpoint].copy(), self.frame[:, midpoint:].copy(),
                               preprocessed=self.input_mode.currentIndex() == 1)

    def _invalidate_frame(self):
        self.frame_revision += 1
        if hasattr(self, 'preview_view'):
            self.preview_view.frame = None
            self.preview_view.refresh()
            self.result_label.setText('等待识别')
            self.color_label.setText('颜色：—　置信度：—')
            self.mask_status_label.setText('左覆膜：未就绪　右覆膜：未就绪')

    @Slot(int, int, object)
    def _on_recognized(self, request_id, frame_revision, result):
        if self.closing or request_id != self.active_request_id:
            return
        self.active_request_id = None
        self.buttons['手动识别'].setEnabled(True)
        if frame_revision != self.frame_revision:
            return
        if result.error:
            self.result_label.setText('识别失败')
            self.status_label.setText(result.error)
            return
        detection = result.detection
        self.result_label.setText('无目标' if detection is None else str(getattr(detection, 'shape', getattr(detection, 'class_name', '类别 %s' % detection.class_id))))
        color = result.color
        self.color_label.setText('颜色：%s　置信度：%s' % (color.name, '%.1f%%' % (color.confidence * 100)) if color else '颜色：未知　置信度：—')
        if self.input_mode.currentIndex() == 1:
            self.color_label.setText('颜色：特征拼图无法验证原始颜色')
        from legacy_vision import surface_summary
        self.mask_status_label.setText(surface_summary(result))
        self.preview_view.show_frame(result.preview_bgr)
        self.status_label.setText('识别完成 · %.1f ms' % result.elapsed_ms)

    @Slot(int, int, str)
    def _on_recognition_failed(self, request_id, frame_revision, message):
        if self.closing or request_id != self.active_request_id:
            return
        self.active_request_id = None
        self.buttons['手动识别'].setEnabled(True)
        if frame_revision != self.frame_revision:
            return
        self.result_label.setText('模型未就绪')
        self.status_label.setText(message)
        self.log('识别失败：' + message)

    def _discard_closed_model_operation(self, unused):
        if self.model_dialog is None:
            self.model_busy = False

    def _worker_thread_finished(self):
        # QThread.finished can precede native thread-local cleanup. Poll on the
        # UI thread until the native thread is joined, without blocking it.
        if self._thread is not None and not self._thread.wait(0):
            QTimer.singleShot(10, self._worker_thread_finished)
            return
        self._thread = None
        if self.closing:
            self._close_ready = True
            self.close()

    def save(self):
        if self.frame is None:
            self.error('请先加载画面再保存样本')
            return
        try:
            ok, encoded = cv2.imencode('.png', self.frame)
            if not ok:
                raise ValueError('图像编码失败')
            self.last_saved = save_sample(ROOT/'data/samples', self.state.source, encoded.tobytes(),
                frame_index=self.source.frame_index, layout=self.layout_mode,
                preprocessed=self.input_mode.currentIndex() == 1,
                width=int(self.frame.shape[1]), height=int(self.frame.shape[0]),
                detection_ready=False, segmentation_ready=False)
            self.log('样本已保存：' + str(self.last_saved))
        except Exception as exc:
            self.error('保存失败：' + str(exc))

    def settings(self):
        dialog = QDialog(self)
        dialog.setWindowTitle('设置与日志 — 离线阶段')
        dialog.resize(720, 460)
        layout = QVBoxLayout(dialog)
        tabs = QTabWidget()
        page = QWidget()
        form = QFormLayout(page)
        combo = QComboBox()
        combo.addItems(['双视图拼接（中线拆分）', '单视图（显示在左侧）'])
        combo.setCurrentText(self.layout_mode)
        form.addRow('媒体布局', combo)
        form.addRow('目标平台', QLabel('Jetson Xavier NX / JetPack 5（板端实测待完成）'))
        form.addRow('相机 / 串口', QLabel('未连接 / 机械输出关闭'))
        form.addRow('形状 / 掩膜模型', QLabel('侧视：26 类旧模型；覆膜默认关闭，候选模型须试跑并确认'))
        path_label = QLabel(str(ROOT/'data/samples'))
        path_label.setWordWrap(True)
        form.addRow('样本保存目录', path_label)
        model_button = QPushButton('模型管理')
        model_button.clicked.connect(lambda: ModelDialog(self).exec() if not self.model_busy and self.active_request_id is None else None)
        form.addRow('模型', model_button)
        tabs.addTab(page, '设置')
        log = QTextEdit()
        log.setReadOnly(True)
        log.setPlainText('\n'.join(self.logs))
        tabs.addTab(log, '运行日志')
        layout.addWidget(tabs)
        buttons = QDialogButtonBox(QBUTTON_OK | QBUTTON_CANCEL)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() == QDIALOG_ACCEPTED:
            self.layout_mode = combo.currentText()
            (ROOT/'data').mkdir(exist_ok=True)
            try:
                (ROOT/'data/ui_settings.json').write_text(json.dumps({
                    'layout': self.layout_mode,
                    'preprocessed': self.input_mode.currentIndex() == 1,
                }, ensure_ascii=False), encoding='utf8')
            except OSError as exc:
                self.error('设置保存失败：' + str(exc))
            if self.frame is not None:
                self.display(self.frame)

    def closeEvent(self, event):
        if self.model_dialog is not None and self.model_dialog.mutating:
            self.model_dialog.status.setText('模型操作进行中，请等待结果')
            event.ignore()
            return
        if not self._close_ready:
            if not self.closing:
                self.closing = True
                if self.model_dialog is not None:
                    self.model_dialog.reject()
                self._thread.cancel_candidate()
                self.timer.stop()
                self.source.close()
                self._thread.shutdown()
            event.ignore()
            return
        self.timer.stop()
        self.source.close()
        super().closeEvent(event)


def main():
    parser = argparse.ArgumentParser(description='Xavier NX视觉操作台：离线媒体回放')
    parser.add_argument('--file', type=Path)
    parser.add_argument('--fullscreen', action='store_true')
    input_type = parser.add_mutually_exclusive_group()
    input_type.add_argument('--preprocessed', action='store_true', help='输入为已处理的 640×320 双摄特征拼图')
    input_type.add_argument('--raw', action='store_true', help='强制使用原始彩色双摄图，覆盖保存的输入类型')
    args = parser.parse_args()
    app = QApplication(sys.argv[:1])
    window = Window()
    saved = ROOT/'data/ui_settings.json'
    if saved.exists():
        try:
            settings = json.loads(saved.read_text(encoding='utf8'))
            value = settings['layout']
            if value in ('双视图拼接（中线拆分）', '单视图（显示在左侧）'):
                window.layout_mode = value
            if settings.get('preprocessed') is True:
                window.input_mode.setCurrentIndex(1)
        except (ValueError, KeyError, OSError):
            window.log('已忽略不可读取的界面设置。')
    if args.preprocessed:
        window.input_mode.setCurrentIndex(1)
    elif args.raw:
        window.input_mode.setCurrentIndex(0)
    if args.preprocessed or args.raw:
        window.layout_mode = '双视图拼接（中线拆分）'
    window.showFullScreen() if args.fullscreen else window.show()
    if args.file:
        window.load_file(args.file)
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
