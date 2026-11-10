"""Physical XY travel limits, independent of the camera tray dimensions."""
import math


X_MIN_MM = 0.0
X_MAX_MM = 260.0
Y_MIN_MM = 0.0
Y_MAX_MM = 230.0
MACHINE_REGION = (X_MIN_MM, Y_MIN_MM, X_MAX_MM, Y_MAX_MM)

STEPS_PER_MM = (40, 40)
PUSH_END_X_MM = 40.0
PARK_MM = (10.0, 10.0)


def position_in_machine(x, y):
    """Return whether two finite numeric coordinates lie within machine travel."""
    return (type(x) in (int, float) and type(y) in (int, float)
            and X_MIN_MM <= x <= X_MAX_MM and Y_MIN_MM <= y <= Y_MAX_MM
            and math.isfinite(x) and math.isfinite(y))
