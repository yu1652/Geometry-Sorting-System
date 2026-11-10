"""Windows-style debug surface for the Linux 20261107 live profile."""

import time

from app import ImageView, QLabel, QPushButton, QTimer, QT_ALIGN_CENTER, Qt, QWidget

try:
    from PySide6.QtWidgets import (QCheckBox, QGridLayout, QGroupBox, QHBoxLayout,
                                   QMenu, QScrollArea, QSizePolicy, QSpinBox, QTextEdit, QVBoxLayout)
    IGNORE = QSizePolicy.Policy.Ignored
    SCROLLBAR_OFF = Qt.ScrollBarPolicy.ScrollBarAlwaysOff
except ImportError:
    from PyQt5.QtWidgets import (QCheckBox, QGridLayout, QGroupBox, QHBoxLayout,
                                 QMenu, QScrollArea, QSizePolicy, QSpinBox, QTextEdit, QVBoxLayout)
    IGNORE = QSizePolicy.Ignored
    SCROLLBAR_OFF = Qt.ScrollBarAlwaysOff


IDLE_STATES = {'disconnected', 'needs_origin', 'armed', 'stopped'}
TIMED_STATES = {'enabling_auto', 'starting', 'running', 'locating', 'wait_scan',
                'recognizing', 'wait_repush', 'wait_done', 'verifying_drop'}

STYLE = '''
QWidget#releaseDebug {background:#f1f3f6;color:#172437;}
QWidget#releaseDebug QGroupBox {
    border:1px solid #d3dbe3;border-radius:5px;margin-top:7px;
    padding-top:5px;font-weight:700;
}
QWidget#releaseDebug QGroupBox::title {subcontrol-origin:margin;left:7px;padding:0 3px;}
QWidget#releaseDebug QLabel#image {background:#111827;color:#cbd5e1;border-radius:4px;}
QWidget#releaseDebug QLabel[debugTile="true"] {
    background:white;border:1px solid #d3dbe3;border-radius:4px;padding:2px 5px;
}
QWidget#releaseDebug QLabel#debugTotal {
    background:#e9fbf1;border:1px solid #57b785;color:#155934;font-weight:700;
}
QWidget#releaseDebug QLabel#debugClock {font-size:17px;font-weight:700;}
QWidget#releaseDebug QPushButton {padding:2px 4px;min-height:16px;}
QWidget#releaseDebug QPushButton#debugStart {background:#159447;color:white;border-color:#12803e;}
QWidget#releaseDebug QPushButton#debugPause {background:#f5ab18;color:#222;border-color:#d99008;}
QWidget#releaseDebug QPushButton#debugStop {background:#ae4538;color:white;border-color:#96372d;}
'''


def _configure(layout, margin=0, spacing=4):
    layout.setContentsMargins(margin, margin, margin, margin)
    layout.setSpacing(spacing)
    return layout


def _group(title, layout_type=QVBoxLayout):
    group = QGroupBox(title)
    return group, _configure(layout_type(group), 4)


def _tile(text, name=None, centered=False):
    label = QLabel(text)
    label.setProperty('debugTile', True)
    if name:
        label.setObjectName(name)
    if centered:
        label.setAlignment(QT_ALIGN_CENTER)
    return label


def _image(title, placeholder, grid, row, col, span=1):
    frame, layout = _group(title)
    view = ImageView(placeholder)
    # A pixmap changes QLabel's size hint. The grid should govern its size.
    view.setSizePolicy(IGNORE, IGNORE)
    layout.addWidget(view, 1)
    grid.addWidget(frame, row, col, 1, span)
    return view


def _log(title, grid, column, span=1):
    frame, layout = _group(title)
    view = QTextEdit()
    view.setReadOnly(True)
    view.document().setMaximumBlockCount(200)
    view.setMinimumHeight(29)
    view.setMaximumHeight(90)
    view.setSizePolicy(IGNORE, IGNORE)
    layout.addWidget(view)
    grid.addWidget(frame, 3, column, 1, span)
    return view


def _button(text, callback, name=None):
    button = QPushButton(text)
    if name:
        button.setObjectName(name)
    button.clicked.connect(callback)
    return button


def _build_controls(window, side):
    window.debug_sequence_label = _tile('序号：—')
    window.debug_item_label = _tile('货物名称：—')
    window.debug_total_label = _tile('分拣成功总数量：0', 'debugTotal')
    for label in (window.debug_sequence_label, window.debug_item_label, window.debug_total_label):
        side.addWidget(label)

    device_controls = _configure(QHBoxLayout())
    device_controls.addWidget(QLabel('设备'))
    camera_settings = _button('相机设置 / 刷新列表', window.camera_settings)
    camera_settings.setToolTip('打开当前相机列表，选择左轴测、右轴测及俯拍相机。')
    device_controls.addWidget(camera_settings, 1)
    device_controls.addWidget(_button('刷新串口', window.refresh_ports))
    side.addLayout(device_controls)

    window.debug_status_label = QLabel('状态：等待设备就绪')
    window.debug_serial_label = QLabel('串口：未连接')
    window.debug_tray_status_label = QLabel('托盘定位：等待检测')
    window.result_label = QLabel('识别结果：等待识别')
    window.result_label.setWordWrap(True)
    window.debug_status_label.setWordWrap(True)
    side.addWidget(window.debug_status_label)
    device_status = _configure(QHBoxLayout())
    for label in (window.debug_serial_label, window.debug_tray_status_label):
        label.setWordWrap(True)
        device_status.addWidget(label, 1)
    side.addLayout(device_status)
    side.addWidget(window.result_label)

    actions = _configure(QHBoxLayout())
    side.addLayout(actions)
    clock_row = _configure(QHBoxLayout())
    side.addLayout(clock_row)
    clock_row.addWidget(QLabel('计时'))
    clock_label = _tile('00:00', 'debugClock', centered=True)
    clock_row.addWidget(clock_label, 1)

    navigation = _configure(QHBoxLayout())
    side.addLayout(navigation)

    xy_box, xy = _group('XY 位置')
    xy_points = _configure(QHBoxLayout())
    xy.addLayout(xy_points)
    window.debug_home_label = _tile('HOME 原点\n未确认', centered=True)
    window.debug_park_label = _tile('PARK 等待点\n未上报', centered=True)
    window.debug_push_end_label = _tile('推料终止线\n未上报', centered=True)
    for label in (window.debug_home_label, window.debug_park_label, window.debug_push_end_label):
        xy_points.addWidget(label, 1)
    window.debug_position_label = QLabel('位置：未上报')
    window.debug_position_label.setWordWrap(True)
    xy.addWidget(window.debug_position_label)
    xy_actions = _configure(QHBoxLayout())
    xy.addLayout(xy_actions)
    xy_actions.addWidget(_button('回 HOME', window.return_home))
    release_motor = _button('释放电机', window.stop)
    release_motor.setToolTip('发送停止，释放后继续运行需重新手动对零。')
    xy_actions.addWidget(release_motor)
    xy_actions.addWidget(_button('手动对零', window.confirm_origin))
    side.addWidget(xy_box)

    loop_row = _configure(QHBoxLayout())
    side.addLayout(loop_row)
    loop_button = QPushButton('循环识别')
    loop_button.setToolTip('仅在设备空闲且相机可用时循环手动识别。')
    loop_button.setCheckable(True)
    loop_row.addWidget(loop_button, 1)
    loop_row.addWidget(QLabel('间隔'))
    interval = QSpinBox()
    interval.setRange(200, 10000)
    interval.setSingleStep(100)
    interval.setValue(500)
    interval.setSuffix(' ms')
    loop_row.addWidget(interval)

    window.calibration_confirmed = QCheckBox('九点标定已在实机逐点核对')
    side.addWidget(window.calibration_confirmed)
    side.addStretch(1)

    loop_timer = QTimer(loop_button)
    loop_timer.setInterval(interval.value())
    interval.valueChanged.connect(loop_timer.setInterval)

    def can_recognize():
        machine = window.machine
        return (window.debug_page.isVisible() and window.vision is not None
                and not window._closing and not window._models_open()
                and not machine.pending
                and machine.snapshot().state in IDLE_STATES
                and (not machine.connected or machine.idle_confirmed))

    def recognize_once():
        if not can_recognize():
            loop_button.setChecked(False)
        elif window._job is None and window._frames_at is not None:
            window.manual_recognize()

    def toggle_loop(enabled):
        if not enabled:
            loop_timer.stop()
            return
        if not can_recognize():
            window.write_log('循环识别需要空闲设备和已连接的相机。')
            loop_button.setChecked(False)
            return
        loop_timer.start()
        recognize_once()

    loop_button.toggled.connect(toggle_loop)
    loop_timer.timeout.connect(recognize_once)

    def end_loop_then(callback):
        loop_button.setChecked(False)
        callback()

    actions.addWidget(_button('运行', lambda: end_loop_then(window.start), 'debugStart'), 1)
    window.debug_pause_button = _button('暂停', window.toggle_feed_pause, 'debugPause')
    window.debug_pause_button.setToolTip('本件完成后暂停进料，保持电机和已确认的零点。')
    window.debug_pause_button.setEnabled(False)
    actions.addWidget(window.debug_pause_button, 1)
    for text, callback, name in (
        ('停止', lambda: end_loop_then(window.stop), 'debugStop'),
        ('手动识别', window.manual_recognize, None),
        ('复位', lambda: end_loop_then(window.reset), None),
        ('XY 标定', window.show_calibration, None),
    ):
        actions.addWidget(_button(text, callback, name), 1)

    navigation.addWidget(_button('切换到比赛界面', lambda: end_loop_then(window.show_competition)), 1)
    navigation.addWidget(_button('本轮箱号', window.binding_dialog))
    more = QPushButton('更多')
    menu = QMenu(more)
    more.setMenu(menu)
    navigation.addWidget(more)
    window.empty_reference_button = _button('采集空区', window.capture_empty_reference)
    loop_row.addWidget(window.empty_reference_button)
    for text, callback in (
        ('托盘定位（只显示）', window.manual_locate),
        ('位置没动，继续', lambda: window.write_log('当前固件在释放电机后需要重新手动对零。')),
        ('清空本轮', lambda: end_loop_then(window.clear_round)),
        ('退出', lambda: end_loop_then(window.close)),
    ):
        menu.addAction(text).triggered.connect(callback)

    elapsed = 0.0
    previous_tick = time.monotonic()
    was_active = False
    clock_timer = QTimer(clock_label)

    def update_clock():
        nonlocal elapsed, previous_tick, was_active
        now = time.monotonic()
        if was_active:
            elapsed += now - previous_tick
        previous_tick = now
        state = window.machine.snapshot().state
        was_active = state in TIMED_STATES
        minutes, seconds = divmod(int(elapsed), 60)
        clock_label.setText(f'{minutes:02d}:{seconds:02d}')

    def reset_clock():
        nonlocal elapsed, previous_tick
        elapsed = 0.0
        previous_tick = time.monotonic()
        update_clock()

    clock_row.addWidget(_button('清零', reset_clock))
    clock_timer.timeout.connect(update_clock)
    clock_timer.start(200)


def build_release_debug(window):
    """Populate the live window's debug page and expose its live-update widgets."""
    page = window.debug_page
    page.setObjectName('releaseDebug')
    page.setStyleSheet(STYLE)
    grid = page.layout()
    if grid is None:
        grid = QGridLayout(page)
    elif not isinstance(grid, QGridLayout):
        body = QWidget(page)
        grid.addWidget(body)
        grid = QGridLayout(body)
    _configure(grid, 3)
    for col, stretch in enumerate((1, 1, 3, 0)):
        grid.setColumnStretch(col, stretch)
    grid.setRowStretch(0, 4)
    grid.setRowStretch(1, 4)
    grid.setRowStretch(3, 1)

    window.views = {}
    window.views['left'] = _image('左轴测实时画面', '等待左相机', grid, 0, 0)
    window.views['right'] = _image('右轴测实时画面', '等待右相机', grid, 0, 1)
    window.preview = _image('拼接检测图', '等待拼接检测', grid, 0, 2)
    window.color_view = _image('颜色检测', '等待颜色检测', grid, 1, 0, 2)
    window.views['top'] = _image('托盘俯拍定位图', '等待托盘定位', grid, 1, 2)

    scroll = QScrollArea(page)
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(SCROLLBAR_OFF)
    scroll.setStyleSheet('QScrollArea {border:0;background:transparent;}')
    sidebar = QWidget()
    _build_controls(window, _configure(QVBoxLayout(sidebar), spacing=3))
    scroll.setWidget(sidebar)
    scroll.setMinimumWidth(315)
    grid.addWidget(scroll, 0, 3, 2, 1)

    bins, bin_row = _group('6 个储物箱绑定关系', QHBoxLayout)
    window.debug_bin_labels = []
    for number in range(1, 7):
        label = _tile(f'{number}号箱 · 未绑定 · 0件', centered=True)
        label.setWordWrap(True)
        bin_row.addWidget(label, 1)
        window.debug_bin_labels.append(label)
    grid.addWidget(bins, 2, 0, 1, 4)

    window.tray_log_view = _log('托盘定位日志', grid, 0, 2)
    window.log_view = _log('识别结果日志', grid, 2)
    window.serial_log_view = _log('上下位机通讯', grid, 3)
