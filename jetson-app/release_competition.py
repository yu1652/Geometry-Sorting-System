"""Compact competition display for the Linux 20261107 profile."""

import cv2

try:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor, QImage, QPixmap
    from PySide6.QtWidgets import (QComboBox, QFrame, QGridLayout, QHBoxLayout,
        QLabel, QPushButton, QSizePolicy, QStackedWidget, QTableWidget, QTableWidgetItem,
        QVBoxLayout, QWidget)
    CENTER = Qt.AlignmentFlag.AlignCenter
    KEEP = Qt.AspectRatioMode.KeepAspectRatio
    SMOOTH = Qt.TransformationMode.SmoothTransformation
    RGB888 = QImage.Format.Format_RGB888
    IGNORE = QSizePolicy.Policy.Ignored
except ImportError:
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QColor, QImage, QPixmap
    from PyQt5.QtWidgets import (QComboBox, QFrame, QGridLayout, QHBoxLayout,
        QLabel, QPushButton, QSizePolicy, QStackedWidget, QTableWidget, QTableWidgetItem,
        QVBoxLayout, QWidget)
    CENTER, KEEP, SMOOTH, RGB888 = Qt.AlignCenter, Qt.KeepAspectRatio, Qt.SmoothTransformation, QImage.Format_RGB888
    IGNORE = QSizePolicy.Ignored


def _label(text, name=None):
    widget = QLabel(str(text))
    if name:
        widget.setObjectName(name)
    widget.setAlignment(CENTER)
    widget.setWordWrap(True)
    return widget


def _button(text, callback):
    widget = QPushButton(text)
    widget.clicked.connect(callback)
    return widget


class _Frame(QLabel):
    def __init__(self, placeholder):
        super().__init__(placeholder)
        self.placeholder = placeholder
        self.frame = None
        self.setAlignment(CENTER)
        self.setMinimumSize(1, 1)
        self.setSizePolicy(IGNORE, IGNORE)
        self.setStyleSheet('background:#e2e8f0;color:#566c85;border-radius:3px;')

    def show_frame(self, frame):
        self.frame = None if frame is None else frame.copy()
        self.refresh()

    def refresh(self):
        if self.frame is None:
            self.clear()
            self.setText(self.placeholder)
            return
        if not self.isVisible() or self.width() < 2 or self.height() < 2:
            return
        rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        image = QImage(rgb.data, width, height, rgb.strides[0], RGB888).copy()
        self.setPixmap(QPixmap.fromImage(image).scaled(self.size(), KEEP, SMOOTH))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh()

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh()


class ReleaseCompetition(QWidget):
    """Three views of one shared round state, with host controls wired to LiveWindow."""

    MODES = ('卡片', '看板', '表格')

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self._total = 0
        self._bins = [('未绑定', 0, None) for _ in range(6)]
        self._status = '等待设备就绪'
        self._current = {}
        self._recognition = self._tray = None
        self.setStyleSheet('''
            ReleaseCompetition {background:#f3f6f8;color:#263747;}
            QFrame#tile {background:#ffffff;border:1px solid #dae3ea;border-radius:8px;}
            QLabel#number {font-size:28px;font-weight:700;color:#2876bc;}
            QLabel#total {font-size:22px;font-weight:700;color:#187f77;background:#dff2e9;border-radius:8px;padding:4px 12px;}
            QLabel#heading {font-size:18px;font-weight:700;}
            QLabel#status {background:#fff;border:1px solid #cfd9e6;border-radius:4px;font-size:14px;font-weight:600;padding:3px;}
            QLabel[summary="true"] {background:#fff;border:1px solid #cfd9e6;border-radius:4px;padding:3px;}
            QTableWidget {background:#fff;gridline-color:#ccd7e4;}
            QHeaderView::section {background:#edf2f7;border:1px solid #ccd7e4;padding:3px;}
            QPushButton {padding:4px 8px;min-height:22px;}
        ''')
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 6, 8, 6)
        outer.setSpacing(5)
        top = QHBoxLayout()
        title = _label('智能分拣系统', 'heading')
        top.addWidget(title, 1)
        self.total_label = _label('分拣成功总数量：0', 'total')
        top.addWidget(self.total_label, 1)
        self.status_line = _label(self._status, 'status')
        top.addWidget(self.status_line, 1)
        outer.addLayout(top)

        self.stack = QStackedWidget()
        outer.addWidget(self.stack, 1)
        self.card_page = QWidget()
        self.board_page = QWidget()
        self.table_page = QWidget()
        for page in (self.card_page, self.board_page, self.table_page):
            self.stack.addWidget(page)
        self._make_cards()
        self._make_board()
        self._make_table()

        self.current_line = _label('当前物件：—')
        self.current_line.setProperty('summary', True)
        outer.addWidget(self.current_line)
        self.preview_bar = QWidget()
        self.preview_bar.setFixedHeight(56)
        previews = QHBoxLayout(self.preview_bar)
        previews.setContentsMargins(0, 0, 0, 0)
        self.small_recognition = _Frame('识别区预览')
        self.small_tray = _Frame('托盘预览')
        self.small_recognition.setFixedHeight(56)
        self.small_tray.setFixedHeight(56)
        previews.addWidget(_label('识别'))
        previews.addWidget(self.small_recognition, 1)
        previews.addWidget(_label('托盘'))
        previews.addWidget(self.small_tray, 1)
        outer.addWidget(self.preview_bar)

        controls = QHBoxLayout()
        controls.addWidget(QLabel('布局'))
        self.mode_picker = QComboBox()
        self.mode_picker.addItems(self.MODES)
        self.mode_picker.currentIndexChanged.connect(self._set_mode)
        controls.addWidget(self.mode_picker)
        controls.addWidget(_button('运行', window.start))
        self.pause_button = _button('暂停', window.toggle_feed_pause)
        self.pause_button.setToolTip('本件完成后暂停进料，保持已确认的零点。')
        self.pause_button.setEnabled(False)
        controls.addWidget(self.pause_button)
        for text, action in (
            ('停止', window.stop), ('复位', window.reset),
            ('零点确认', window.confirm_origin),
            ('相机设置', window.camera_settings), ('标定', window.show_calibration),
            ('切换到调试界面', window.show_debug), ('退出', window.close),
        ):
            controls.addWidget(_button(text, action))
        outer.addLayout(controls)
        self.render(0, self._bins, self._status, {})

    def _tile(self, title, count, preview, parent):
        tile = QFrame(parent)
        tile.setObjectName('tile')
        body = QVBoxLayout(tile)
        body.setContentsMargins(8, 5, 8, 6)
        body.setSpacing(3)
        heading = _label(title, 'heading')
        body.addWidget(preview, 1)
        body.addWidget(heading)
        amount = _label(count, 'number')
        body.addWidget(amount)
        return tile, heading, amount

    def _make_cards(self):
        grid = QGridLayout(self.card_page)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(6)
        self.card_tiles = []
        for index in range(6):
            view = _Frame('暂无图片')
            tile, heading, amount = self._tile(f'{index + 1}号箱 · 未绑定', '0 件', view, self.card_page)
            grid.addWidget(tile, index // 3, index % 3)
            self.card_tiles.append((heading, amount, view))

    def _make_board(self):
        body = QHBoxLayout(self.board_page)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(6)
        left = QFrame()
        left.setObjectName('tile')
        column = QVBoxLayout(left)
        column.addWidget(QLabel('当前分拣'))
        details = QGridLayout()
        self.board_fields = {}
        for index, (key, label) in enumerate((
            ('sequence', '序号'), ('name', '货物名称'), ('bin', '储物盒'),
            ('count', '当前类别数量'), ('serial', '串口'), ('tray', '托盘定位'),
            ('color', '颜色'), ('shape', '形状'),
        )):
            value = _label(label + '：—')
            value.setProperty('summary', True)
            details.addWidget(value, index // 2, index % 2)
            self.board_fields[key] = (value, label)
        column.addLayout(details)
        previews = QHBoxLayout()
        self.recognition_view = _Frame('识别预览')
        self.tray_view = _Frame('托盘预览')
        previews.addWidget(self.recognition_view)
        previews.addWidget(self.tray_view)
        column.addLayout(previews, 1)
        body.addWidget(left, 2)
        right = QWidget()
        grid = QGridLayout(right)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(5)
        self.board_tiles = []
        for index in range(6):
            tile = QFrame()
            tile.setObjectName('tile')
            layout = QVBoxLayout(tile)
            layout.setContentsMargins(5, 4, 5, 4)
            heading = _label(f'{index + 1}号箱')
            image = _Frame('暂无图片')
            amount = _label('0', 'number')
            layout.addWidget(image, 1)
            layout.addWidget(heading)
            layout.addWidget(amount)
            grid.addWidget(tile, index // 3, index % 3)
            self.board_tiles.append((heading, amount, image))
        body.addWidget(right, 3)

    def _make_table(self):
        layout = QVBoxLayout(self.table_page)
        layout.setContentsMargins(0, 0, 0, 0)
        self.table = QTableWidget(4, 6)
        self.table.horizontalHeader().hide()
        self.table.setVerticalHeaderLabels(('储物盒序号', '货物名称', '货物图片', '成功数量'))
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers if hasattr(QTableWidget, 'EditTrigger') else QTableWidget.NoEditTriggers)
        header = self.table.horizontalHeader()
        stretch = header.ResizeMode.Stretch if hasattr(header, 'ResizeMode') else header.Stretch
        header.setSectionResizeMode(stretch)
        self.table.setRowHeight(0, 30)
        self.table.setRowHeight(1, 32)
        self.table.setRowHeight(3, 35)
        self.table.verticalHeader().setSectionResizeMode(2, stretch)
        self.table_images = []
        for column in range(6):
            image = _Frame('暂无图片')
            self.table.setCellWidget(2, column, image)
            self.table_images.append(image)
        layout.addWidget(self.table, 1)

    def _set_mode(self, index):
        self.stack.setCurrentIndex(index)
        self._paint_visible()

    def render(self, total, bins, status, current):
        if len(bins) != 6:
            raise ValueError('bins must contain six entries')
        self._total = total
        self._bins = list(bins)
        self._status = str(status)
        self._current = dict(current or {})
        self.total_label.setText(f'分拣成功总数量：{total}')
        self._paint_visible()

    def set_frames(self, recognition_bgr, tray_bgr):
        self._recognition = None if recognition_bgr is None else recognition_bgr.copy()
        self._tray = None if tray_bgr is None else tray_bgr.copy()
        self.small_recognition.show_frame(self._recognition)
        self.small_tray.show_frame(self._tray)
        if self.stack.currentWidget() is self.board_page or recognition_bgr is None or tray_bgr is None:
            self.recognition_view.show_frame(self._recognition)
            self.tray_view.show_frame(self._tray)

    def _paint_visible(self):
        current = self._current
        detail = '当前物件：{}  ·  箱号 {}  ·  序号 {}  ·  颜色 {}  ·  形状 {}'.format(
            current.get('name', '—'), current.get('bin', '—'),
            current.get('sequence', '—'), current.get('color', '—'), current.get('shape', '—'))
        self.status_line.setText(self._status)
        self.current_line.setText(
            '{}  ·  本箱 {}  ·  串口 {}  ·  托盘 {}'.format(
                detail, current.get('count', '—'), current.get('serial', '—'),
                current.get('tray', '—')))
        page = self.stack.currentWidget()
        board = page is self.board_page
        self.preview_bar.setVisible(not board)
        self.current_line.setVisible(not board)
        if page is self.card_page:
            for index, (name, count, frame) in enumerate(self._bins):
                heading, amount, view = self.card_tiles[index]
                heading.setText(f'{index + 1}号箱 · {name}')
                amount.setText(f'{count} 件')
                view.show_frame(frame)
        elif page is self.board_page:
            for key, (widget, label) in self.board_fields.items():
                widget.setText('{}：{}'.format(label, current.get(key, '—')))
            for index, (name, count, frame) in enumerate(self._bins):
                heading, amount, image = self.board_tiles[index]
                heading.setText(f'{index + 1}号箱 · {name}')
                amount.setText(str(count))
                image.show_frame(frame)
            self.recognition_view.show_frame(self._recognition)
            self.tray_view.show_frame(self._tray)
        else:
            for column, (name, count, frame) in enumerate(self._bins):
                for row, value in ((0, column + 1), (1, name), (3, count)):
                    item = QTableWidgetItem(str(value))
                    item.setTextAlignment(CENTER)
                    if row in (0, 3):
                        item.setForeground(QColor('#2161ed'))
                    self.table.setItem(row, column, item)
                self.table_images[column].show_frame(frame)
