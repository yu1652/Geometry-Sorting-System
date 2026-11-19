"""Real ESP32 protocol controller. Only received MCU events acknowledge actions."""
import math
import time
from collections import deque

from motion_limits import X_MAX_MM, Y_MAX_MM, position_in_machine
from sorter_simulation import SorterSnapshot


MCU_STATES = {'WAIT_START', 'IDLE', 'WAIT_PUSH', 'PUSHING', 'WAIT_SORT', 'SORTING', 'STOP'}
REJECT_STATES = {'reject_opening', 'reject_holding', 'reject_closing', 'reject_settling'}
SUPPORTED_MCU_VERSIONS = {'0.1.0', '0.1.0-stop.1'}
# Match DROP_HOLD_MS and CLOSE_SETTLE_MS in the paired ESP32 firmware.
REJECT_GATE_OPEN_SECONDS = 1.0
REJECT_GATE_SETTLE_SECONDS = 1.0
REJECT_GATE_ACK_TIMEOUT = 5.0
AUTO_STATES = {'enabling_auto', 'starting', 'running', 'locating', 'wait_scan', 'recognizing',
               'wait_repush', 'wait_done', 'verifying_drop', 'paused'} | REJECT_STATES


def _position(x, y):
    return position_in_machine(x, y)


class LiveController:
    def __init__(self, send, clock=time.monotonic, timeout=30, steps_per_mm=(40, 39),
                 verify_drop=False, scan_timeout=None, done_timeout=None, accept_boot_origin=False):
        if not callable(send) or not callable(clock):
            raise ValueError('send and clock must be callable')
        if type(accept_boot_origin) is not bool:
            raise ValueError('accept_boot_origin must be a boolean')
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('timeout must be finite and positive')
        for name, value in (('scan_timeout', scan_timeout), ('done_timeout', done_timeout)):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(name + ' must be finite and positive')
        if len(steps_per_mm) != 2 or any(not math.isfinite(v) or v <= 0 for v in steps_per_mm):
            raise ValueError('steps_per_mm must contain two positive finite values')
        self._send, self._clock, self._timeout = send, clock, timeout
        self._verify_drop = verify_drop
        self._phase_timeouts = {state: value for state, value in
                                (('recognizing', scan_timeout), ('wait_done', done_timeout))
                                if value is not None}
        self._steps = tuple(steps_per_mm)
        self._connected = False
        self.ready = False
        self.epoch = 0
        self._state = 'disconnected'
        self._origin = False
        self._pending = None
        self._target = None
        self._bin = None
        self._x = self._y = 0.0
        self._pusher_down = False
        self._feed_pause_requested = False
        self._counts = [0] * 6
        self._log = deque(maxlen=200)
        self._last_error = None
        self._boot_id = None
        self._accept_boot_origin = accept_boot_origin
        self._boot_origin_seen = None
        self._mcu_state = None
        self._got_pong = self._got_version = self._got_status = False
        self._deadline = None
        self._last_rx = clock()
        # A normal STOP forgets motion authority, not the coordinate anchor.
        # Reusing it still requires the operator to confirm no physical change.
        self._known_origin_boot = None
        self._reset_phase = None
        self._reset_boot = None
        self._reset_command = None
        self._reset_echo = False
        self._reset_status_echo = False
        self._reset_position = None
        self._reset_clear_counts = False
        self._reset_generation = 0
        self._reject_resume_at = None
        self._reject_command_echo = False
        self._reject_gate_may_be_open = False

    @property
    def connected(self):
        return self._connected

    @property
    def pending(self):
        return self._pending

    @property
    def idle_confirmed(self):
        """A real MCU idle/stop report, with no manual or automatic action pending."""
        return (self.ready and self._mcu_state in ('WAIT_START', 'STOP')
                and self._state in ('needs_origin', 'stopped', 'armed')
                and self._pending is None and not self._pusher_down)

    @property
    def feed_pause_requested(self):
        return self._feed_pause_requested

    @property
    def request_epoch(self):
        return self.epoch

    @property
    def can_reuse_origin(self):
        return (self._connected and self.ready and not self.reset_in_progress
                and self._pending is None and self._known_origin_boot is not None
                and self._known_origin_boot == self._boot_id)

    @property
    def reset_in_progress(self):
        return self._reset_phase is not None

    @property
    def reset_generation(self):
        return self._reset_generation

    def snapshot(self):
        return SorterSnapshot(self._state, self._origin, self._x, self._y,
                              self._pusher_down, sum(self._counts), tuple(self._counts),
                              tuple(self._log), self._last_error)

    def _write(self, line):
        if not self._connected:
            return False
        try:
            self._send(line)
        except Exception as error:
            self.disconnect('串口写入失败：' + str(error))
            return False
        self._log.append('HOST ' + line)
        return True

    def _reject(self, reason):
        self._last_error = reason
        self._log.append('HOST ' + reason)
        return False

    def _disarm(self, state):
        self._cancel_reject()
        self.epoch += 1
        self._origin = False
        self._feed_pause_requested = False
        self._pending = self._target = self._bin = None
        self._deadline = None
        self._state = state

    def _wait(self, state):
        self._state = state
        duration = self._phase_timeouts.get(state, self._timeout)
        self._deadline = self._clock() + duration

    def _expired_action(self):
        if ((self._verify_drop or self._phase_timeouts or self.reset_in_progress
             or self._state in REJECT_STATES or self._state == 'wait_repush')
                and self._deadline is not None and self._clock() >= self._deadline):
            self._fault(self._timeout_reason())
            self.ready = False
            return True
        return False

    def confirm_drop(self, epoch):
        """Accept a fresh host clearance result after the MCU's DONE event."""
        if self._state == 'verifying_drop' and self._expired_action():
            return False
        if (not self._verify_drop or self._state != 'verifying_drop'
                or not self.ready or not self._origin or self._bin is None
                or epoch != self.epoch):
            return False
        self._counts[self._bin - 1] += 1
        self._bin = None
        self._wait('running')
        return True

    def begin_handshake(self):
        self._cancel_known_reset(invalidate=True)
        self._disarm('handshaking')
        self._connected = True
        self.ready = False
        self._got_pong = self._got_version = self._got_status = False
        self._last_rx = self._clock()
        self._wait('handshaking')
        for command in ('PING', 'VERSION?', 'STATUS?'):
            if not self._write(command):
                return False
        return True

    def disconnect(self, reason='串口已断开'):
        if self._reject_gate_may_be_open:
            reason += '；底板关闭未确认，请现场检查'
        self._cancel_known_reset(invalidate=True)
        self._disarm('disconnected')
        self._connected = self.ready = False
        self._last_error = reason
        self._log.append('HOST ' + reason)

    def confirm_origin(self):
        if not self.ready or self._state not in ('needs_origin', 'stopped'):
            return self._reject('需要通信就绪且设备停机，手动放到零点后确认')
        if self._mcu_state not in ('WAIT_START', 'STOP'):
            return self._reject('MCU 尚未停机，请先停止并读取状态')
        self._known_origin_boot = None
        self._disarm('resetting')
        self._last_error = None
        self._wait('resetting')
        return self._write('RESET')

    def start(self):
        if not self.ready or not self._origin or self._state != 'armed' or self._pusher_down or self.pending:
            return self._reject('启动需要原点确认、推板抬起且手动动作已结束')
        self._last_error = None
        self._wait('enabling_auto')
        return self._write('HOST_READY')

    def set_feed_paused(self, paused):
        if not self.ready or not self._origin or self._state not in AUTO_STATES:
            return self._reject('自动分拣中才可暂停或继续进料')
        self._feed_pause_requested = paused
        if self._state == 'paused' and not paused:
            self._wait('running')
        return True

    def stop(self):
        close_gate = self._reject_gate_may_be_open
        self._cancel_known_reset(invalidate=self.reset_in_progress)
        self._disarm('stopped')
        self._mcu_state = None
        if not self._connected:
            self._state = 'disconnected'
            return False
        # The existing firmware STOP only raises the pusher/disables XY. Close
        # an unknown-part gate explicitly before cancelling the host sequence.
        closed = self._write('SERVO2,CLOSE') if close_gate else True
        stopped = self._write('STOP') and self._write('STATUS?')
        return closed and stopped

    def invalidate_origin(self, reason='零点或机构位置已改变'):
        """Stop after an explicit position change without inventing a new zero."""
        self._known_origin_boot = None
        self._last_error = reason
        return self.stop()

    def reset_known_origin(self, clear_counts=True):
        """Return to an unchanged, same-boot anchor before rebasing counters.

        This uses only existing firmware commands. Every stage waits for its
        real RX/event/status sequence; an ACK is never fabricated locally.
        """
        if type(clear_counts) is not bool:
            return self._reject('清空计数参数必须为布尔值')
        if not self.can_reuse_origin:
            return self._reject('缺少本次控制板启动期间的可信零点，或已有动作未结束')
        self._reset_boot = self._boot_id
        self._reset_clear_counts = clear_counts
        self._last_error = None
        self._disarm('reset_stopping')
        return self._known_reset_stage('stopping', 'STOP')

    def _cancel_known_reset(self, invalidate=False):
        if invalidate:
            self._known_origin_boot = None
        self._reset_phase = self._reset_boot = self._reset_command = None
        self._reset_echo = self._reset_status_echo = False
        self._reset_position = None
        self._reset_clear_counts = False

    def _known_reset_stage(self, phase, command):
        self._reset_phase = phase
        self._reset_command = command
        self._reset_echo = self._reset_status_echo = False
        self._wait('reset_' + phase)
        return self._write(command)

    def _known_reset_event(self, line, fields):
        if fields[0] == 'RX':
            command = line[3:]
            if command == self._reset_command:
                self._reset_echo = True
            if self._reset_phase == 'homing' and self._reset_echo and command == 'STATUS?':
                self._reset_status_echo = True
            return False
        if not self._reset_echo:
            return False
        if self._reset_phase == 'stopping' and fields == ['STATUS', 'STOP']:
            self._mcu_state = 'STOP'
            self._known_reset_stage('stopped_status', 'STATUS?')
            return True
        if self._reset_phase == 'lifting' and fields == ['STATUS', 'PUSHER_UP']:
            self._pusher_down = False
            if self._reset_position == (0, 0):
                self._known_reset_stage('resetting', 'RESET')
            elif self._known_reset_stage('homing', 'GOTO_MM,0.000,0.000'):
                self._write('STATUS?')
            return True
        if self._reset_phase == 'resetting' and fields == ['READY']:
            self._known_reset_stage('verifying', 'STATUS?')
            return True
        return False

    def _known_reset_status(self, fields, pulses):
        if fields[5] != self._reset_boot:
            self._boot_id = fields[5]
            self.begin_handshake()
            self._last_error = '控制板已重新启动，原零点不能自动沿用'
            return True
        self._mcu_state = fields[2]
        self._x, self._y = (pulses[i] / self._steps[i] for i in range(2))
        self._last_rx = self._clock()
        if not self._reset_echo:
            return True
        if self._reset_phase == 'stopped_status' and fields[2] == 'STOP':
            if not _position(self._x, self._y):
                self._fault('复位前坐标超出机械范围，无法自动回零')
                return True
            self._reset_position = pulses
            self._known_reset_stage('lifting', 'PUSHER,UP')
        elif (self._reset_phase == 'homing' and self._reset_status_echo
              and fields[2] == 'STOP' and pulses == (0, 0)):
            self._known_reset_stage('resetting', 'RESET')
        elif (self._reset_phase == 'verifying' and fields[2] == 'WAIT_START'
              and pulses == (0, 0)):
            boot, clear_counts = self._reset_boot, self._reset_clear_counts
            self._cancel_known_reset()
            self._known_origin_boot = boot
            self._origin = True
            self._pusher_down = False
            self._pending = self._target = self._bin = None
            self._state = 'armed'
            self._deadline = None
            if clear_counts:
                self._counts = [0] * 6
            self._reset_generation += 1
            self._log.append('HOST 完全复位已确认：HOME 0,0，待启动')
        return True

    def goto(self, x, y):
        if not self.ready or not self._origin or self._state != 'armed' or self._pusher_down or self.pending:
            return self._reject('移动需要停机、原点确认、推板抬起且无待完成动作')
        if not _position(x, y):
            return self._reject(f'XY 必须为 X 0 到 {X_MAX_MM:g}、Y 0 到 {Y_MAX_MM:g} mm 内的有限数字')
        self._target = (float('{:.3f}'.format(x)), float('{:.3f}'.format(y)))
        self._pending = 'goto'
        self._deadline = self._clock() + self._timeout
        return self._write('GOTO_MM,{:.3f},{:.3f}'.format(x, y)) and self._write('STATUS?')

    def set_pusher(self, down):
        if type(down) is not bool or not self.ready or not self._connected:
            return self._reject('推板动作需要通信就绪和有效状态')
        if down:
            if not self._origin or self._state != 'armed' or self.pending:
                return self._reject('下降需要停机、原点确认且无待完成动作')
        elif self._pending == 'goto':
            # The GOTO handler already raises before XY motion; keep waiting for its STATUS.
            return self._write('PUSHER,UP')
        elif self._state not in ('armed', 'stopped', 'needs_origin'):
            # Avoid an UP queued during automatic pushing changing a later movement phase.
            return self._reject('自动动作期间请使用停止，随后再抬起')
        self._pending = 'down' if down else 'up'
        self._deadline = self._clock() + self._timeout
        return self._write('PUSHER,DOWN' if down else 'PUSHER,UP')

    def provide_target(self, x=None, y=None):
        if self._state == 'locating' and self._expired_action():
            return False
        if not self.ready or not self._origin or self._state != 'locating':
            return self._reject('没有等待中的定位请求')
        if x is None and y is None:
            self._wait('running')
            return self._write('PUSH,WAIT')
        if not _position(x, y) or float('{:.3f}'.format(x)) <= 10:
            self._fault('推料坐标越界或方向无效')
            return False
        self._wait('wait_scan')
        return self._write('PUSH,{:.3f},{:.3f},mm'.format(x, y))

    def provide_sort(self, bin_id):
        if self._state == 'recognizing' and self._expired_action():
            return False
        if not self.ready or not self._origin or self._state != 'recognizing':
            return self._reject('没有等待中的识别请求')
        if type(bin_id) is not int or not 1 <= bin_id <= 6:
            self._fault('无有效箱号绑定')
            return False
        self._bin = bin_id
        self._wait('wait_done')
        return self._write('SORT,{}'.format(bin_id))

    def retry_empty_scan(self):
        """Return an independently confirmed empty station to the push loop.

        Existing firmware accepts PUSH,WAIT during WAIT_SORT, schedules its next
        PUSH? and performs no gate/bin action. Wait for a real ACK/request;
        never invent DONE or reuse the previous tray coordinate.
        """
        if self._state == 'recognizing' and self._expired_action():
            return False
        if not self.ready or not self._origin or self._state != 'recognizing':
            return self._reject('当前没有可重新推料的识别请求')
        self.epoch += 1
        self._bin = None
        self._wait('wait_repush')
        return self._write('PUSH,WAIT')

    def reject_unknown_scan(self):
        """Dispose of this unclassified part before permitting another push.

        Servo commands leave the MCU in WAIT_SORT. Only their own real RX and
        gate replies authorize the timed holds; no SORT, BIN or DONE is used.
        """
        if self._state == 'recognizing' and self._expired_action():
            return False
        if (not self._connected or not self.ready or not self._origin
                or self._state != 'recognizing' or self.pending or self._bin is not None):
            return self._reject('当前没有可排出的未知料识别请求')
        self._reject_gate_may_be_open = True
        return self._reject_gate_command('reject_opening', 'SERVO2,OPEN')

    def _cancel_reject(self):
        self._reject_resume_at = None
        self._reject_command_echo = False

    def _reject_gate_command(self, state, command):
        self._cancel_reject()
        self._state = state
        self._deadline = self._clock() + REJECT_GATE_ACK_TIMEOUT
        return self._write(command)

    def _hold_reject_gate(self, state, duration):
        self._state = state
        self._reject_resume_at = self._clock() + duration
        self._deadline = self._reject_resume_at + REJECT_GATE_ACK_TIMEOUT
        self._reject_command_echo = False

    def _fault(self, reason):
        self._cancel_known_reset(invalidate=True)
        self._last_error = reason
        self._log.append('HOST ' + reason)
        self.stop()

    def _timeout_reason(self):
        if self._clock() - self._last_rx >= self._timeout:
            return '串口通信超时：未收到有效回执，请检查连接后重新确认零点'
        if self._state == 'recognizing':
            seconds = self._phase_timeouts.get('recognizing', self._timeout)
            return '识别复核超时（限时 {:g} 秒）：请检查相机和识别日志后重新确认零点'.format(seconds)
        if self._state == 'wait_done':
            return '投箱完成等待超时：未收到 DONE 回执，请检查机构后重新确认零点'
        return '动作等待超时（阶段 {}）：请检查机构后重新确认零点'.format(self._state)

    def tick(self):
        now = self._clock()
        if self._connected and (self.ready or self._state == 'handshaking') and (
            (self._deadline is not None and now >= self._deadline)
            or now - self._last_rx >= self._timeout
        ):
            self._fault(self._timeout_reason())
            self.ready = False
            return
        if (not self._connected or not self.ready or not self._origin
                or self._reject_resume_at is None or now < self._reject_resume_at):
            return
        if self._state == 'reject_holding':
            self._reject_gate_command('reject_closing', 'SERVO2,CLOSE')
        elif self._state == 'reject_settling':
            self._cancel_reject()
            self.epoch += 1
            self._bin = None
            self._wait('wait_repush')
            self._write('PUSH,WAIT')

    def handle_mcu(self, line):
        if not self._connected or not isinstance(line, str):
            return False
        line = line.strip()
        if not line or len(line) > 256 or any(not 32 <= ord(c) <= 126 for c in line):
            return False
        self._log.append('MCU ' + line)
        fields = line.split(',')
        head = fields[0]
        if head == 'RX':
            if self._state in ('reject_opening', 'reject_closing'):
                if self._expired_action():
                    return False
                expected = 'RX,SERVO2,OPEN' if self._state == 'reject_opening' else 'RX,SERVO2,CLOSE'
                if line == expected:
                    self._reject_command_echo = True
            if self.reset_in_progress:
                return self._known_reset_event(line, fields)
            return False
        if head == 'ERROR':
            self._last_rx = self._clock()
            self._fault(line)
            return True
        if head == 'BOOT' and len(fields) == 2 and fields[1].isdigit():
            self._boot_id = fields[1]
            return self.begin_handshake()
        if fields == ['STATUS', 'ORIGIN_REQUIRED']:
            self._cancel_known_reset(invalidate=True)
            self._disarm('stopped')
            self._last_error = 'XY 运动中途停止，请将机构放回标定零点再确认。'
            self._last_rx = self._clock()
            return True
        if self.reset_in_progress and fields[:2] != ['STATUS', 'MCU']:
            if self._expired_action():
                return False
            handled = self._known_reset_event(line, fields)
            if handled:
                self._last_rx = self._clock()
            return handled
        if line in ('READY', 'PUSH?', 'SCAN', 'DONE') and self._expired_action():
            return False
        if line == 'READY':
            self._last_rx = self._clock()
            if self._state == 'resetting':
                self._x = self._y = 0.0
                self._mcu_state = 'WAIT_START'
                self._origin = True
                self._known_origin_boot = self._boot_id
                self._pusher_down = False
                self._state = 'armed'
                self._deadline = None
                return True
            return self.begin_handshake()
        if line == 'PONG':
            self._got_pong = True
        elif len(fields) == 3 and fields[:2] == ['VERSION', 'ESP32_SORTER_XY_PUSHER']:
            if self._state != 'handshaking':
                return False
            if fields[2] not in SUPPORTED_MCU_VERSIONS:
                self.disconnect('固件版本 {} 不匹配，请使用 Y 轴 39 脉冲/mm 的配套 0.1.0 或 0.1.0-stop.1。'.format(fields[2]))
                return True
            self._got_version = True
        elif fields[:2] == ['STATUS', 'MCU']:
            return self._mcu_status(fields)
        elif head == 'STATUS':
            return self._status(fields[1:])
        elif line == 'PUSH?' and self.ready and self._origin and self._state in ('running', 'wait_repush', 'paused'):
            if self._feed_pause_requested:
                self._wait('paused')
                if not self._write('PUSH,WAIT'):
                    return False
            else:
                self.epoch += 1
                self._wait('locating')
        elif line == 'SCAN' and self.ready and self._origin and self._state == 'wait_scan':
            self.epoch += 1
            self._wait('recognizing')
        elif line == 'DONE' and self._state == 'wait_done' and self._bin is not None:
            if self._verify_drop:
                self.epoch += 1
                self._wait('verifying_drop')
            else:
                self._counts[self._bin - 1] += 1
                self._bin = None
                self._wait('running')
        else:
            return False
        self._last_rx = self._clock()
        self._finish_handshake()
        return True

    def _finish_handshake(self):
        if self._state == 'handshaking' and self._got_pong and self._got_version and self._got_status:
            self.ready = True
            self._state = 'needs_origin'
            self._deadline = None
            self._last_error = None
            if (self._accept_boot_origin and self._boot_id is not None
                    and self._boot_origin_seen != self._boot_id):
                self._boot_origin_seen = self._boot_id
                if self._mcu_state == 'WAIT_START' and (self._x, self._y) == (0.0, 0.0):
                    self._origin = True
                    self._known_origin_boot = self._boot_id
                    self._pusher_down = False
                    self._state = 'armed'
                    self._log.append('HOST 已采用 ESP32 启动位置作为本次软件零点（0,0），未执行回零移动')

    def _mcu_status(self, fields):
        if len(fields) != 6 or fields[2] not in MCU_STATES or not fields[5].isdigit():
            return False
        try:
            pulses = tuple(int(value) for value in fields[3:5])
        except ValueError:
            return False
        if self.reset_in_progress:
            if self._expired_action():
                return False
            return self._known_reset_status(fields, pulses)
        if self._boot_id is not None and fields[5] != self._boot_id:
            self.begin_handshake()
        self._boot_id = fields[5]
        self._mcu_state = fields[2]
        self._x, self._y = (pulses[i] / self._steps[i] for i in range(2))
        self._last_rx = self._clock()
        self._got_status = True
        self._finish_handshake()
        if self._state in REJECT_STATES and self._mcu_state != 'WAIT_SORT':
            self._fault('排出未知料期间控制板状态改变，已停止')
            return True
        if self._mcu_state == 'STOP' and self._state not in ('handshaking', 'needs_origin', 'resetting'):
            self._disarm('stopped')
        elif self._pending == 'goto' and self._mcu_state == 'WAIT_START':
            expected = tuple(math.floor(self._target[i] * self._steps[i] + 0.5) for i in range(2))
            if pulses == expected:
                self._pending = self._target = None
                self._deadline = None
            # A heartbeat STATUS may have been queued before GOTO. A mismatched
            # position does not acknowledge arrival or extend the action deadline.
        return True

    def _status(self, fields):
        if self._state in REJECT_STATES and self._expired_action():
            return False
        if fields in (['HOST_READY'], ['START'], ['PUSH', 'WAIT']) and self._expired_action():
            return False
        self._last_rx = self._clock()
        if (fields == ['SORT_GATE_OPEN'] and self._state == 'reject_opening'
                and self._reject_command_echo):
            self._hold_reject_gate('reject_holding', REJECT_GATE_OPEN_SECONDS)
            return True
        if (fields == ['SORT_GATE_CLOSE'] and self._state == 'reject_closing'
                and self._reject_command_echo):
            self._reject_gate_may_be_open = False
            self._hold_reject_gate('reject_settling', REJECT_GATE_SETTLE_SECONDS)
            return True
        if fields == ['PUSH', 'WAIT'] and self.ready and self._origin and self._state == 'wait_repush':
            self._wait('running')
            return True
        if fields == ['HOST_READY'] and self._state == 'enabling_auto':
            self._wait('starting')
            return self._write('START')
        if fields == ['START']:
            if not self.ready or not self._origin or self._state != 'starting':
                self._fault('MCU 在自动流程未获授权时启动，请检查后重新手动对零')
                return False
            if self._pusher_down or self.pending:
                self._fault('未完成调试动作时 MCU 意外启动')
                return False
            self._wait('running')
            return True
        if fields == ['STOP']:
            self._mcu_state = 'STOP'
            self._disarm('stopped')
            if self._reject_gate_may_be_open:
                self._write('SERVO2,CLOSE')
            return True
        if fields in (['PUSHER_UP'], ['PUSHER_DOWN']):
            down = fields == ['PUSHER_DOWN']
            self._pusher_down = down
            if self._pending == ('down' if down else 'up'):
                self._pending = None
                self._deadline = None
            return True
        return False
