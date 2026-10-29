#include "xy_motion.h"

#include "comm.h"
#include "config.h"

static long currentX = HOME_X_PULSE;
static long currentY = HOME_Y_PULSE;

static long roundedSteps(float value) {
    return value >= 0.0f ? static_cast<long>(value + 0.5f) : static_cast<long>(value - 0.5f);
}

static unsigned long minStepIntervalUs() {
    return static_cast<unsigned long>(XY_STEP_HIGH_US) + static_cast<unsigned long>(XY_STEP_LOW_US);
}

static unsigned long clampedStepInterval(unsigned long intervalUs) {
    unsigned long minimum = minStepIntervalUs();
    return intervalUs < minimum ? minimum : intervalUs;
}

static bool readActiveLevel(int pin, int activeLevel) {
    if (pin < 0) {
        return false;
    }
    return digitalRead(pin) == activeLevel;
}

static void pulseStepPin(int pin, unsigned int intervalUs) {
    unsigned long stepIntervalUs = clampedStepInterval(intervalUs);
    unsigned long lowDelayUs = stepIntervalUs > XY_STEP_HIGH_US
        ? stepIntervalUs - XY_STEP_HIGH_US
        : XY_STEP_LOW_US;
    if (lowDelayUs < XY_STEP_LOW_US) {
        lowDelayUs = XY_STEP_LOW_US;
    }

    digitalWrite(pin, HIGH);
    delayMicroseconds(XY_STEP_HIGH_US);
    digitalWrite(pin, LOW);
    delayMicroseconds(lowDelayUs);
}

static bool homeAxis(
    const char *axisName,
    int stepPin,
    int dirPin,
    int limitPin,
    int homeDirLevel,
    long backoffSteps,
    long maxTravelSteps
) {
    if (limitPin < 0) {
        sendText(String("ERROR,HOME_NO_LIMIT_PIN,") + axisName);
        return false;
    }

    digitalWrite(dirPin, !homeDirLevel);
    for (long step = 0; step < backoffSteps && readActiveLevel(limitPin, HOME_LIMIT_ACTIVE_LEVEL); ++step) {
        pulseStepPin(stepPin, HOME_BACKOFF_STEP_INTERVAL_US);
    }
    if (readActiveLevel(limitPin, HOME_LIMIT_ACTIVE_LEVEL)) {
        sendText(String("ERROR,HOME_LIMIT_STUCK,") + axisName);
        return false;
    }

    digitalWrite(dirPin, homeDirLevel);
    bool triggered = readActiveLevel(limitPin, HOME_LIMIT_ACTIVE_LEVEL);
    for (long step = 0; step < maxTravelSteps && !triggered; ++step) {
        pulseStepPin(stepPin, HOME_FAST_STEP_INTERVAL_US);
        triggered = readActiveLevel(limitPin, HOME_LIMIT_ACTIVE_LEVEL);
    }
    if (!triggered) {
        sendText(String("ERROR,HOME_FAST_FAIL,") + axisName);
        return false;
    }

    digitalWrite(dirPin, !homeDirLevel);
    for (long step = 0; step < backoffSteps; ++step) {
        pulseStepPin(stepPin, HOME_BACKOFF_STEP_INTERVAL_US);
    }
    delay(80);
    if (readActiveLevel(limitPin, HOME_LIMIT_ACTIVE_LEVEL)) {
        sendText(String("ERROR,HOME_LIMIT_STUCK,") + axisName);
        return false;
    }

    digitalWrite(dirPin, homeDirLevel);
    triggered = readActiveLevel(limitPin, HOME_LIMIT_ACTIVE_LEVEL);
    for (long step = 0; step < backoffSteps * 3 && !triggered; ++step) {
        pulseStepPin(stepPin, HOME_SLOW_STEP_INTERVAL_US);
        triggered = readActiveLevel(limitPin, HOME_LIMIT_ACTIVE_LEVEL);
    }
    if (!triggered) {
        sendText(String("ERROR,HOME_SLOW_FAIL,") + axisName);
        return false;
    }

    sendText(String("STATUS,HOME_AXIS_DONE,") + axisName);
    return true;
}

static unsigned long profileStepIntervalUs(
    long stepIndex,
    long totalSteps,
    unsigned long configuredStartInterval,
    unsigned long configuredCruiseInterval,
    long configuredAccelSteps
) {
    if (!XY_ACCEL_ENABLE || totalSteps <= 1 || configuredAccelSteps <= 0) {
        return clampedStepInterval(configuredCruiseInterval);
    }

    unsigned long startInterval = clampedStepInterval(configuredStartInterval);
    unsigned long cruiseInterval = clampedStepInterval(configuredCruiseInterval);
    if (startInterval <= cruiseInterval) {
        return cruiseInterval;
    }

    long rampSteps = min(configuredAccelSteps, totalSteps / 2);
    if (rampSteps <= 0) {
        return cruiseInterval;
    }

    long distanceFromStart = stepIndex + 1;
    long distanceToEnd = totalSteps - stepIndex;
    long rampPosition = min(distanceFromStart, distanceToEnd);
    if (rampPosition >= rampSteps) {
        return cruiseInterval;
    }

    float ratio = static_cast<float>(rampPosition) / static_cast<float>(rampSteps);
    float startSpeed = 1000000.0f / static_cast<float>(startInterval);
    float cruiseSpeed = 1000000.0f / static_cast<float>(cruiseInterval);
    float speed = startSpeed + (cruiseSpeed - startSpeed) * ratio;
    if (speed <= 0.0f) {
        return startInterval;
    }
    return clampedStepInterval(static_cast<unsigned long>((1000000.0f / speed) + 0.5f));
}

static void writeEnablePin(int pin, bool enabled) {
    if (pin < 0) {
        return;
    }
    digitalWrite(pin, enabled ? XY_ENABLE_ACTIVE_LEVEL : !XY_ENABLE_ACTIVE_LEVEL);
}

void initXYMotion() {
    pinMode(X_STEP_PIN, OUTPUT);
    pinMode(X_DIR_PIN, OUTPUT);
    if (X_ENABLE_PIN >= 0) {
        pinMode(X_ENABLE_PIN, OUTPUT);
    }

    pinMode(Y_STEP_PIN, OUTPUT);
    pinMode(Y_DIR_PIN, OUTPUT);
    if (Y_ENABLE_PIN >= 0) {
        pinMode(Y_ENABLE_PIN, OUTPUT);
    }
    if (USE_HOME_LIMIT_SWITCHES) {
        if (X_HOME_LIMIT_PIN >= 0) {
            pinMode(X_HOME_LIMIT_PIN, HOME_LIMIT_USE_PULLUP ? INPUT_PULLUP : INPUT);
        }
        if (Y_HOME_LIMIT_PIN >= 0) {
            pinMode(Y_HOME_LIMIT_PIN, HOME_LIMIT_USE_PULLUP ? INPUT_PULLUP : INPUT);
        }
    }

    digitalWrite(X_STEP_PIN, LOW);
    digitalWrite(Y_STEP_PIN, LOW);
    enableXYMotors(true);
    xySetCurrentPosition(HOME_X_PULSE, HOME_Y_PULSE);
}

void enableXYMotors(bool enabled) {
    writeEnablePin(X_ENABLE_PIN, enabled);
    writeEnablePin(Y_ENABLE_PIN, enabled);
}

bool xyHome() {
    if (!USE_HOME_LIMIT_SWITCHES) {
        return xyMoveTo(HOME_X_PULSE, HOME_Y_PULSE);
    }

    sendText("STATUS,HOME_START");
    enableXYMotors(true);

    long backoffX = max(xyMmToPulseX(HOME_BACKOFF_MM), 1L);
    long backoffY = max(xyMmToPulseY(HOME_BACKOFF_MM), 1L);
    long maxTravelX = max(xyMmToPulseX(HOME_MAX_TRAVEL_MM), backoffX + 1);
    long maxTravelY = max(xyMmToPulseY(HOME_MAX_TRAVEL_MM), backoffY + 1);

    if (!homeAxis("X", X_STEP_PIN, X_DIR_PIN, X_HOME_LIMIT_PIN, X_HOME_DIR_LEVEL, backoffX, maxTravelX)) {
        return false;
    }
    if (!homeAxis("Y", Y_STEP_PIN, Y_DIR_PIN, Y_HOME_LIMIT_PIN, Y_HOME_DIR_LEVEL, backoffY, maxTravelY)) {
        return false;
    }

    xySetCurrentPosition(HOME_X_PULSE, HOME_Y_PULSE);
    sendText(String("STATUS,HOME_DONE,") + String(currentX) + "," + String(currentY));
    return true;
}

bool xyInSoftLimit(long x, long y) {
    return x >= X_MIN_PULSE && x <= X_MAX_PULSE && y >= Y_MIN_PULSE && y <= Y_MAX_PULSE;
}

bool xyInSoftLimitMm(float xMm, float yMm) {
    return xyInSoftLimit(xyMmToPulseX(xMm), xyMmToPulseY(yMm));
}

bool xyInTrayRangeMm(float xMm, float yMm) {
    return xMm >= 0.0f && xMm <= TRAY_WIDTH_MM && yMm >= 0.0f && yMm <= TRAY_HEIGHT_MM;
}

long xyMmToPulseX(float xMm) {
    return roundedSteps(xMm * X_STEPS_PER_MM);
}

long xyMmToPulseY(float yMm) {
    return roundedSteps(yMm * Y_STEPS_PER_MM);
}

bool xyMoveToMm(float targetXmm, float targetYmm) {
    if (!xyInSoftLimitMm(targetXmm, targetYmm)) {
        sendText(String("ERROR,XY_MM_SOFT_LIMIT,") + String(targetXmm, 2) + "," + String(targetYmm, 2));
        return false;
    }
    return xyMoveTo(xyMmToPulseX(targetXmm), xyMmToPulseY(targetYmm));
}

static bool xyMoveToProfile(
    long targetX,
    long targetY,
    unsigned long startIntervalUs,
    unsigned long cruiseIntervalUs,
    long accelSteps
) {
    if (!xyInSoftLimit(targetX, targetY)) {
        sendText(String("ERROR,XY_SOFT_LIMIT,") + String(targetX) + "," + String(targetY));
        return false;
    }

    long dx = targetX - currentX;
    long dy = targetY - currentY;
    long stepsX = labs(dx);
    long stepsY = labs(dy);
    long totalSteps = max(stepsX, stepsY);

    if (totalSteps == 0) {
        return true;
    }

    digitalWrite(X_DIR_PIN, dx >= 0 ? X_DIR_POSITIVE_LEVEL : !X_DIR_POSITIVE_LEVEL);
    digitalWrite(Y_DIR_PIN, dy >= 0 ? Y_DIR_POSITIVE_LEVEL : !Y_DIR_POSITIVE_LEVEL);
    enableXYMotors(true);

    long accX = 0;
    long accY = 0;
    for (long step = 0; step < totalSteps; ++step) {
        bool pulseX = false;
        bool pulseY = false;

        accX += stepsX;
        if (accX >= totalSteps) {
            accX -= totalSteps;
            pulseX = true;
        }

        accY += stepsY;
        if (accY >= totalSteps) {
            accY -= totalSteps;
            pulseY = true;
        }

        if (pulseX) {
            digitalWrite(X_STEP_PIN, HIGH);
        }
        if (pulseY) {
            digitalWrite(Y_STEP_PIN, HIGH);
        }
        unsigned long stepIntervalUs = profileStepIntervalUs(
            step,
            totalSteps,
            startIntervalUs,
            cruiseIntervalUs,
            accelSteps
        );
        unsigned long lowDelayUs = stepIntervalUs > XY_STEP_HIGH_US
            ? stepIntervalUs - XY_STEP_HIGH_US
            : XY_STEP_LOW_US;
        if (lowDelayUs < XY_STEP_LOW_US) {
            lowDelayUs = XY_STEP_LOW_US;
        }

        delayMicroseconds(XY_STEP_HIGH_US);
        if (pulseX) {
            digitalWrite(X_STEP_PIN, LOW);
            currentX += dx >= 0 ? 1 : -1;
        }
        if (pulseY) {
            digitalWrite(Y_STEP_PIN, LOW);
            currentY += dy >= 0 ? 1 : -1;
        }
        delayMicroseconds(lowDelayUs);
    }

    currentX = targetX;
    currentY = targetY;

    if (!XY_HOLD_ENABLE) {
        enableXYMotors(false);
    }
    sendText(String("STATUS,XY_AT,") + String(currentX) + "," + String(currentY));
    return true;
}

bool xyMoveTo(long targetX, long targetY) {
    return xyMoveToProfile(
        targetX,
        targetY,
        XY_START_STEP_INTERVAL_US,
        XY_CRUISE_STEP_INTERVAL_US,
        XY_ACCEL_STEPS
    );
}

bool xyPushToMm(float targetXmm, float targetYmm) {
    if (!xyInSoftLimitMm(targetXmm, targetYmm)) {
        sendText(String("ERROR,PUSH_MM_SOFT_LIMIT,") + String(targetXmm, 2) + "," + String(targetYmm, 2));
        return false;
    }
    return xyMoveToProfile(
        xyMmToPulseX(targetXmm),
        xyMmToPulseY(targetYmm),
        PUSH_START_STEP_INTERVAL_US,
        PUSH_CRUISE_STEP_INTERVAL_US,
        PUSH_ACCEL_STEPS
    );
}

void xySetCurrentPosition(long x, long y) {
    currentX = x;
    currentY = y;
}

long xyCurrentX() {
    return currentX;
}

long xyCurrentY() {
    return currentY;
}