"""Deterministic, in-process model of the sorter protocol for the console."""

from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple


from motion_limits import X_MAX_MM, Y_MAX_MM, position_in_machine


MAX_X_MM = X_MAX_MM
MAX_Y_MM = Y_MAX_MM
PUSH_END_X_MM = 10.0
PARK_MM = (49.0, 5.0)


@dataclass(frozen=True)
class SorterSnapshot:
    state: str
    origin_confirmed: bool
    x_mm: float
    y_mm: float
    pusher_down: bool
    completed_count: int
    bin_counts: Tuple[int, ...]
    log: Tuple[str, ...]
    last_error: Optional[str]


def _in_envelope(x, y):
    return position_in_machine(x, y)


class SimulatedSorter:
    def __init__(self):
        self._state = "needs_origin"
        self._x_mm = self._y_mm = 0.0
        self._pusher_down = False
        self._bin_counts = [0] * 6
        self._log = deque(maxlen=200)
        self._last_error = None
        self._boot_id = None
        self._pending_push = None
        self._pending_bin = None
        self.handle_mcu("READY")

    def snapshot(self):
        return SorterSnapshot(
            self._state,
            self._state not in ("needs_origin", "stopped"),
            self._x_mm,
            self._y_mm,
            self._pusher_down,
            sum(self._bin_counts),
            tuple(self._bin_counts),
            tuple(self._log),
            self._last_error,
        )

    def _host(self, command):
        self._log.append("HOST " + command)

    def _reject(self, reason):
        self._last_error = reason
        self._log.append("SIM " + reason)
        return False

    def _disarm(self, state):
        self._state = state
        self._pusher_down = False
        self._pending_push = self._pending_bin = None

    def confirm_origin(self):
        if self._state not in ("needs_origin", "stopped"):
            return self._reject("Origin confirmation requires a stopped or disarmed sorter")
        if self._state == "stopped":
            self._host("RESET")
            self.handle_mcu("READY")
        self._x_mm = self._y_mm = 0.0
        self._last_error = None
        self._state = "armed"
        self._host("HOST_READY")
        return True

    def start(self):
        if self._state != "armed" or self._pusher_down:
            return self._reject("Start requires confirmed origin and raised pusher")
        self._last_error = None
        return self.handle_mcu("STATUS,START")

    def stop(self):
        self._host("STOP")
        return self.handle_mcu("STATUS,STOP")

    def goto(self, x, y):
        if self._state != "armed" or self._pusher_down:
            return self._reject("Manual XY requires maintenance access and raised pusher")
        if not _in_envelope(x, y):
            return self._reject("Manual XY is outside the simulated motion envelope")
        self._x_mm, self._y_mm = float(x), float(y)
        self._host(f"GOTO_MM,{x},{y}")
        self._last_error = None
        return True

    def set_pusher(self, down):
        if type(down) is not bool:
            return self._reject("Pusher state must be a boolean")
        if down and self._state != "armed":
            return self._reject("Lowering the pusher requires maintenance access")
        if self._state == "armed":
            self._host("PUSHER,DOWN" if down else "PUSHER,UP")
        self._pusher_down = down
        return True

    def begin_cycle(self, x_mm, y_mm, bin_id=None):
        """Simulate PUSH? and leave the cycle waiting for SCAN."""
        if self._state != "running" or self._pusher_down:
            return self._reject("Automatic cycle requires running state and raised pusher")
        valid = _in_envelope(x_mm, y_mm) and x_mm > PUSH_END_X_MM
        if valid:
            x_wire, y_wire = f"{x_mm:.1f}", f"{y_mm:.1f}"
            valid = _in_envelope(float(x_wire), float(y_wire)) and float(x_wire) > PUSH_END_X_MM
        if not valid:
            self._pending_push = "PUSH,NG"
            self.handle_mcu("PUSH?")
            return self._reject("Push point is outside the simulated envelope or direction")
        self._pending_push = f"PUSH,{x_wire},{y_wire},mm"
        self._pending_bin = bin_id
        self._last_error = None
        return self.handle_mcu("PUSH?")

    def run_cycle(self, x_mm, y_mm, bin_id):
        return (
            self.begin_cycle(x_mm, y_mm, bin_id)
            and self.handle_mcu("SCAN")
            and self.handle_mcu("DONE")
        )

    def complete_scan(self, bin_id):
        if self._state != "wait_scan":
            return self._reject("No simulated scan is pending")
        self._pending_bin = bin_id
        return self.handle_mcu("SCAN")

    def handle_mcu(self, line):
        """Consume one simulated wire event and return whether it was handled."""
        line = line.strip()
        if not line:
            return False
        self._log.append("MCU " + line)
        head, *fields = line.split(",")
        if head in ("READY", "BOOT"):
            if head == "BOOT":
                self._boot_id = fields[0] if fields else None
            self._disarm("needs_origin")
            return True
        if head == "ERROR":
            self._last_error = line
            self._host("STOP")
            self._disarm("stopped")
            return True
        if head == "STATUS":
            return self._handle_status(fields)
        if fields:
            return False
        if head == "PUSH?" and self._state in ("running", "wait_scan"):
            if self._pending_push is None:
                self._host("PUSH,WAIT")
            elif self._pending_push == "PUSH,NG":
                self._host(self._pending_push)
                self._pending_push = None
            else:
                self._host(self._pending_push)
                self._state = "wait_scan"
            return True
        if head == "SCAN" and self._state == "wait_scan":
            return self._sort()
        if head == "DONE" and self._state == "wait_done":
            self._bin_counts[self._pending_bin - 1] += 1
            self._pending_bin = None
            self._state = "running"
            return True
        return False

    def _handle_status(self, fields):
        if fields == ["START"] and self._state == "armed" and not self._pusher_down:
            self._state = "running"
            return True
        if fields[:1] == ["STOP"]:
            self._disarm("stopped")
            return True
        if fields[:1] == ["MCU"]:
            boot_id = fields[4] if len(fields) >= 5 else None
            if self._boot_id is not None and boot_id != self._boot_id:
                self._disarm("needs_origin")
            self._boot_id = boot_id
            return True
        return False

    def _sort(self):
        self._pending_push = None
        self._x_mm, self._y_mm = PARK_MM
        bin_id = self._pending_bin
        if type(bin_id) is not int or not 1 <= bin_id <= 6:
            self.handle_mcu("ERROR,INVALID_BIN")
            return False
        self._host(f"SORT,{bin_id}")
        self._state = "wait_done"
        return True
