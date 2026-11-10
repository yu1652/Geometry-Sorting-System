"""Real ESP32 protocol controller. Only received MCU events acknowledge actions."""
import math
import re
import time
from collections import deque

from motion_limits import X_MAX_MM, Y_MAX_MM, position_in_machine
from sorter_simulation import SorterSnapshot


MCU_STATES = {'WAIT_START', 'IDLE', 'WAIT_PUSH', 'PUSHING', 'WAIT_SORT', 'SORTING', 'STOP'}


def _position(x, y):
    return position_in_machine(x, y)


class LiveController:
    def __init__(self, send, clock=time.monotonic, timeout=30, steps_per_mm=(40, 39),
                 verify_drop=False, scan_timeout=None):
        if not callable(send) or not callable(clock):
            raise ValueError('send and clock must be callable')
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('timeout must be finite and positive')
        if scan_timeout is not None and (not math.isfinite(scan_timeout) or scan_timeout <= 0):
            raise ValueError('scan_timeout must be finite and positive')
        if len(steps_per_mm) != 2 or any(not math.isfinite(v) or v <= 0 for v in steps_per_mm):
            raise ValueError('steps_per_mm must contain two positive finite values')
        self._send, self._clock, self._timeout = send, clock, timeout
        self._verify_drop = verify_drop
        self._scan_timeout = scan_timeout
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
        self._counts = [0] * 6
        self._log = deque(maxlen=200)
        self._last_error = None
        self._boot_id = None
        self._mcu_state = None
        self._got_pong = self._got_version = self._got_status = False
        self._deadline = None
        self._last_rx = clock()

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
    def request_epoch(self):
        return self.epoch

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
        self.epoch += 1
        self._origin = False
        self._pending = self._target = self._bin = None
        self._deadline = None
        self._state = state

    def _wait(self, state):
        self._state = state
        duration = self._scan_timeout if state == 'recognizing' and self._scan_timeout is not None else self._timeout
        self._deadline = self._clock() + duration

    def _expired_action(self):
        if ((self._verify_drop or self._scan_timeout is not None)
                and self._deadline is not None and self._clock() >= self._deadline):
            self._fault('通信或动作等待超时，请检查机构后重新确认零点')
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
        self._disarm('disconnected')
        self._connected = self.ready = False
        self._last_error = reason
        self._log.append('HOST ' + reason)

    def confirm_origin(self):
        if not self.ready or self._state not in ('needs_origin', 'stopped'):
            return self._reject('需要通信就绪且设备停机，手动放到零点后确认')
        if self._mcu_state not in ('WAIT_START', 'STOP'):
            return self._reject('MCU 尚未停机，请先停止并读取状态')
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

    def stop(self):
        self._disarm('stopped')
        self._mcu_state = None
        if not self._connected:
            self._state = 'disconnected'
            return False
        return self._write('STOP') and self._write('STATUS?')

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

    def _fault(self, reason):
        self._last_error = reason
        self._log.append('HOST ' + reason)
        self.stop()

    def tick(self):
        now = self._clock()
        if self._connected and (self.ready or self._state == 'handshaking') and (
            (self._deadline is not None and now >= self._deadline)
            or now - self._last_rx >= self._timeout
        ):
            self._fault('通信或动作等待超时，请检查机构后重新确认零点')
            self.ready = False

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
            return False
        if head == 'ERROR':
            self._last_rx = self._clock()
            self._fault(line)
            return True
        if head == 'BOOT' and len(fields) == 2 and fields[1].isdigit():
            self._boot_id = fields[1]
            return self.begin_handshake()
        if line in ('READY', 'PUSH?', 'SCAN', 'DONE') and self._expired_action():
            return False
        if line == 'READY':
            self._last_rx = self._clock()
            if self._state == 'resetting':
                self._x = self._y = 0.0
                self._mcu_state = 'WAIT_START'
                self._origin = True
                self._pusher_down = False
                self._state = 'armed'
                self._deadline = None
                return True
            return self.begin_handshake()
        if line == 'PONG':
            self._got_pong = True
        elif (len(fields) == 3 and fields[:2] == ['VERSION', 'ESP32_SORTER_XY_PUSHER']
              and re.fullmatch(r'\d+\.\d+\.\d+', fields[2])):
            self._got_version = True
        elif fields[:2] == ['STATUS', 'MCU']:
            return self._mcu_status(fields)
        elif head == 'STATUS':
            return self._status(fields[1:])
        elif line == 'PUSH?' and self.ready and self._origin and self._state in ('running', 'wait_repush'):
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

    def _mcu_status(self, fields):
        if len(fields) != 6 or fields[2] not in MCU_STATES or not fields[5].isdigit():
            return False
        try:
            pulses = tuple(int(value) for value in fields[3:5])
        except ValueError:
            return False
        if self._boot_id is not None and fields[5] != self._boot_id:
            self.begin_handshake()
        self._boot_id = fields[5]
        self._mcu_state = fields[2]
        self._x, self._y = (pulses[i] / self._steps[i] for i in range(2))
        self._last_rx = self._clock()
        self._got_status = True
        self._finish_handshake()
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
        if fields in (['HOST_READY'], ['START']) and self._expired_action():
            return False
        self._last_rx = self._clock()
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
            return True
        if fields in (['PUSHER_UP'], ['PUSHER_DOWN']):
            down = fields == ['PUSHER_DOWN']
            self._pusher_down = down
            if self._pending == ('down' if down else 'up'):
                self._pending = None
                self._deadline = None
            return True
        return False
