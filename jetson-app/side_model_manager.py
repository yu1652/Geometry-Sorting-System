"""Shape model manager launched by the full live console after disconnecting IO."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from app import exec_qt
from app import (QApplication, QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                 QPushButton, QComboBox, QFileDialog, QMessageBox, STYLE, Qt)
try:
    from PySide6.QtCore import QObject, QThread, QTimer, Signal
    from PySide6.QtWidgets import QPlainTextEdit
except ImportError:
    from PyQt5.QtCore import QObject, QThread, QTimer, pyqtSignal as Signal
    from PyQt5.QtWidgets import QPlainTextEdit

from side_model_store import SideModelStore, DEFAULT_SAVED_APP


class ReceiverBridge(QObject):
    received = Signal(object, str, str)
    event = Signal(object)


class PrepareWorker(QThread):
    progress = Signal(str)
    ready = Signal(object)
    failed = Signal(str)

    def __init__(self, store, source, mode, reuse, parent=None):
        super().__init__(parent)
        self.store, self.source, self.mode, self.reuse = store, source, mode, reuse

    def run(self):
        try:
            if self.reuse == 'rollback':
                result = self.store.rollback_previous(progress=self.progress.emit)
            else:
                method = self.store.prepare_existing if self.reuse else self.store.prepare_onnx
                result = method(self.source, self.mode, self.progress.emit)
            self.ready.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc) or type(exc).__name__)


class SideModelDialog(QDialog):
    def __init__(self, store=None, receiver=None, enable_transfer=False):
        super().__init__()
        self.store = store or SideModelStore()
        self.worker = None
        self.candidate = None
        self.receiver = receiver
        self.transfer_job = None
        self.rollback_pending = False
        self.started_at = None
        self.receive_until = 0
        self.closing = False
        self.bridge = ReceiverBridge(self)
        self.bridge.received.connect(self.on_received)
        self.bridge.event.connect(self.on_transfer_event)
        self.setStyleSheet(STYLE)
        self.setWindowTitle('ONNX 模型转换器 · Jetson')
        self.resize(780, 620)
        layout = QVBoxLayout(self)
        active = self.store.read_active()
        text = ('当前：%d类 / 320输入 / %s' % (len(active['class_names']), active['preprocess'])) if active else '当前：原启动参数指定的模型'
        self.active_label = QLabel(text)
        layout.addWidget(self.active_label)
        self.mode = QComboBox()
        self.mode.addItems(['灰度 + CLAHE + 边缘（当前 8 类模型使用）', '原始彩色 RGB（仅适用于这样训练的模型）'])
        layout.addWidget(self.mode)
        note = QLabel('① 选择本机 ONNX，或允许 Windows 发送　② 自动校验、复用或转换　③ 确认启用\n'
                      '支持单视图 YOLO：输入 320×320、FP32、无内置 NMS；输入模式必须与训练一致。\n'
                      '新模型首次转换可能需要数分钟；同模型可复用已有 Engine。启用后重启上位机。')
        note.setWordWrap(True)
        layout.addWidget(note)
        row = QHBoxLayout()
        self.reuse_button = QPushButton('从旧测试程序导入 Engine')
        self.reuse_button.clicked.connect(lambda: self.prepare(DEFAULT_SAVED_APP, True))
        row.addWidget(self.reuse_button)
        self.choose_button = QPushButton('选择 ONNX 并转换…')
        self.choose_button.clicked.connect(self.choose)
        row.addWidget(self.choose_button)
        layout.addLayout(row)
        self.receive_button = QPushButton('允许 Windows 发送并转换（5 分钟 / 一份文件）')
        self.receive_button.clicked.connect(self.allow_receive)
        layout.addWidget(self.receive_button)
        self.receive_label = QLabel('未开放接收。两端需要连接同一个局域网。')
        self.receive_label.setWordWrap(True)
        layout.addWidget(self.receive_label)
        self.status = QLabel('请选择模型。候选通过 Engine 试推理后，才能确认启用。')
        self.status.setWordWrap(True)
        layout.addWidget(self.status, 1)
        self.elapsed = QLabel('尚未开始转换')
        layout.addWidget(self.elapsed)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(400)
        self.log.setMinimumHeight(110)
        layout.addWidget(self.log)
        self.confirm_button = QPushButton('确认启用（重启完整上位机后生效）')
        self.confirm_button.setEnabled(False)
        self.confirm_button.clicked.connect(self.activate)
        layout.addWidget(self.confirm_button)
        self.rollback_button = QPushButton('恢复上一个已启用模型')
        self.rollback_button.clicked.connect(self.rollback)
        layout.addWidget(self.rollback_button)
        self.close_button = QPushButton('关闭')
        self.close_button.clicked.connect(self.close)
        layout.addWidget(self.close_button)
        if enable_transfer and self.receiver is None:
            try:
                from model_transfer import ModelReceiver
                self.receiver = ModelReceiver(self.store.root / 'models' / 'inbox',
                                              self.bridge.received.emit, self.bridge.event.emit)
                self.receiver.start()
            except Exception as exc:
                if self.receiver is not None:
                    self.receiver.close()
                self.receiver = None
                self.receive_label.setText('局域网接收未启动：' + str(exc))
        self.receive_button.setEnabled(self.receiver is not None)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(1000)

    def allow_receive(self):
        if self.closing or self.worker is not None or self.receiver is None:
            return
        try:
            self.receiver.allow(seconds=300)
            self.receive_until = time.monotonic() + 300
            self.receive_label.setText('已允许 Windows 发送：5 分钟内接收一份 ONNX。请在 Windows 点“查找 Jetson”。')
        except Exception as exc:
            self.receive_label.setText('无法开放接收：' + str(exc))

    def on_transfer_event(self, event):
        if not self.closing and isinstance(event, dict) and event.get('message'):
            self.receive_label.setText(str(event['message']))

    def on_received(self, path, mode, job_id):
        if self.closing or self.receiver is None:
            return
        self.receive_until = 0
        if self.worker is not None or mode not in ('legacy_features', 'raw_rgb'):
            self.receiver.update_job(job_id, 'failed', '转换器正忙，或输入模式不受支持；请稍后重新允许发送。')
            return
        self.transfer_job = job_id
        self.mode.setCurrentIndex(0 if mode == 'legacy_features' else 1)
        self.receive_label.setText('ONNX 已收到；正在板子上校验和转换。')
        self.receiver.update_job(job_id, 'building', '文件已接收，正在校验和转换。')
        self.prepare(Path(path), False)

    def tick(self):
        if self.started_at is not None and self.worker is not None:
            seconds = int(time.monotonic() - self.started_at)
            self.elapsed.setText('已用时 %02d:%02d · 正在工作，请保留窗口；详细 TensorRT 日志保存在模型版本目录。' % divmod(seconds, 60))
        if self.receive_until:
            seconds = max(0, int(self.receive_until - time.monotonic()))
            self.receive_label.setText('允许接收剩余 %d 秒（仅一份 ONNX）；Windows 点击“查找 Jetson”后发送。' % seconds)
            if seconds == 0:
                self.receive_until = 0
                self.receive_label.setText('接收时间已结束，需要发送时请重新点“允许 Windows 发送”。')

    def on_progress(self, message):
        self.status.setText(message)
        self.log.appendPlainText(message)
        if self.transfer_job and self.receiver:
            self.receiver.update_job(self.transfer_job, 'building', message)

    def choose(self):
        if self.worker is not None:
            return
        path, _ = QFileDialog.getOpenFileName(self, '选择 320×320 单视图 YOLO ONNX', '', 'ONNX (*.onnx)')
        if path:
            self.prepare(Path(path), False)

    def prepare(self, source, reuse):
        if self.closing or self.worker is not None:
            return
        self.candidate = None
        self.confirm_button.setEnabled(False)
        for control in (self.reuse_button, self.choose_button, self.mode, self.close_button, self.receive_button, self.rollback_button):
            control.setEnabled(False)
        self.status.setText('正在准备模型…')
        self.log.clear()
        self.started_at = time.monotonic()
        self.elapsed.setText('已开始校验模型…')
        mode = 'legacy_features' if self.mode.currentIndex() == 0 else 'raw_rgb'
        self.worker = PrepareWorker(self.store, source, mode, reuse, self)
        self.worker.progress.connect(self.on_progress)
        self.worker.ready.connect(self.on_ready)
        self.worker.failed.connect(self.on_failed)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def on_ready(self, candidate):
        if self.rollback_pending:
            self.active_label.setText('已恢复上一个模型（上位机重启后读取）')
            self.status.setText('已校验并恢复上一个模型。请重启完整上位机。')
            self.rollback_pending = False
            return
        self.candidate = candidate
        names = candidate.manifest['class_names']
        self.status.setText('格式及 Engine 试推理通过：%d 类，输入 320，输出 [1,%d,2100]\n类别顺序：%s\n需实拍核对识别效果；请确认训练输入模式后启用。' %
                            (len(names), 4 + len(names), '、'.join(names)))
        self.log.appendPlainText('候选模型：' + str(candidate.manifest_path.parent))
        if self.transfer_job and self.receiver:
            self.receiver.update_job(self.transfer_job, 'ready', '转换及试推理完成，请在 Jetson 点击“确认启用”，然后重启上位机。',
                                     {'class_names': names, 'preprocess': candidate.manifest['preprocess']})
            self.transfer_job = None

    def on_failed(self, message):
        self.status.setText('模型操作失败，当前模型未改变：' + message)
        self.log.appendPlainText(message)
        self.rollback_pending = False
        if self.transfer_job and self.receiver:
            self.receiver.update_job(self.transfer_job, 'failed', message)
            self.transfer_job = None

    def on_finished(self):
        # The finished signal may precede native thread teardown.
        self.worker.wait()
        worker, self.worker = self.worker, None
        worker.deleteLater()
        for control in (self.reuse_button, self.choose_button, self.close_button, self.rollback_button):
            control.setEnabled(True)
        # Keep the mode fixed while a candidate is awaiting confirmation.
        self.mode.setEnabled(self.candidate is None)
        self.confirm_button.setEnabled(self.candidate is not None)
        self.receive_button.setEnabled(self.receiver is not None)
        seconds = int(time.monotonic() - self.started_at) if self.started_at is not None else 0
        self.elapsed.setText('本次操作用时 %02d:%02d' % divmod(seconds, 60))

    def rollback(self):
        if self.worker is not None:
            return
        yes = getattr(QMessageBox, 'Yes', None)
        if yes is None:
            yes = QMessageBox.StandardButton.Yes
        answer = QMessageBox.question(self, '恢复上一个模型', '恢复上一个已启用的模型？校验通过后会切换配置，重启上位机生效。')
        if answer == yes:
            self.rollback_pending = True
            self.prepare(None, 'rollback')

    def activate(self):
        if self.worker is not None or self.candidate is None:
            return
        try:
            self.store.activate(self.candidate)
        except Exception as exc:
            self.status.setText('启用失败，当前模型未改变：' + str(exc))
            return
        self.active_label.setText('已启用 %d 类模型（完整上位机重启后读取）' % len(self.candidate.manifest['class_names']))
        self.status.setText('配置已保存。请关闭本窗口，退出并重启完整上位机。')
        self.candidate = None
        self.confirm_button.setEnabled(False)
        self.mode.setEnabled(True)

    def closeEvent(self, event):
        if self.worker is not None:
            event.ignore()
            return
        self.shutdown_receiver()
        super().closeEvent(event)

    def reject(self):
        if self.worker is None:
            self.shutdown_receiver()
            super().reject()

    def shutdown_receiver(self):
        self.closing = True
        self.timer.stop()
        if self.receiver is not None:
            self.receiver.close()


def standalone_live_processes(root):
    """Avoid opening a second native GPU context alongside the live console.

    The existing in-app entry disconnects IO and vision before launching this
    child. Only that parent is exempt; the desktop launcher asks the user to
    close a separately running console.
    """
    processes = []
    if not Path('/proc').is_dir():
        return processes
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getppid():
            continue
        try:
            args = (proc / 'cmdline').read_bytes().split(b'\0')
            cwd = (proc / 'cwd').resolve()
            if any(Path(os.fsdecode(arg)).name == 'live_console.py' and
                   ((Path(os.fsdecode(arg)).is_absolute() and Path(os.fsdecode(arg)).parent.resolve() == root.resolve())
                    or cwd == root.resolve()) for arg in args if arg):
                processes.append(int(proc.name))
        except (OSError, ValueError):
            pass
    return processes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reuse-saved', action='store_true')
    parser.add_argument('--activate', action='store_true')
    parser.add_argument('--saved-app', type=Path, default=DEFAULT_SAVED_APP)
    args = parser.parse_args()
    if args.reuse_saved:
        store = SideModelStore()
        candidate = store.prepare_existing(args.saved_app, progress=print)
        if args.activate:
            store.activate(candidate)
            print('正式上位机已启用 %d 类，重启后生效；无需重新构建 Engine。' % len(candidate.manifest['class_names']))
        return 0
    application = QApplication(sys.argv[:1])
    if standalone_live_processes(Path(__file__).resolve().parent):
        QMessageBox.information(None, '请先关闭上位机', '请先从上位机“更多→退出”关闭上位机，再打开转换器；也可在上位机点击“模型管理”。')
        return 1
    dialog = SideModelDialog(enable_transfer=True)
    # The live console runs full screen and can cover this window on the Jetson
    # desktop; an invisible manager still blocks device reconnection.
    try:
        dialog.setWindowFlag(Qt.WindowStaysOnTopHint, True)
    except (AttributeError, TypeError):
        pass
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()
    return exec_qt(application)


if __name__ == '__main__':
    raise SystemExit(main())
