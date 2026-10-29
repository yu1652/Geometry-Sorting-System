#include "servos.h"

#include <Arduino.h>
#include <ESP32Servo.h>
#include <Preferences.h>

#include "comm.h"
#include "config.h"

static Servo servo2;
static Servo servo3;
static int currentBinId = SERVO3_DEFAULT_BIN_ID;
static Preferences servoStatePrefs;

static int binAngle(int binId) {
    if (binId < 1 || binId > 6) {
        return BIN_ANGLES[SERVO3_DEFAULT_BIN_ID];
    }
    return constrain(BIN_ANGLES[binId], 0, 180);
}

static void saveCurrentBinId() {
    servoStatePrefs.putInt("last_bin", currentBinId);
}

static unsigned long servo3MoveWaitMs(int fromBinId, int toBinId) {
    int fromAngle = binAngle(fromBinId);
    int toAngle = binAngle(toBinId);
    unsigned long commandDelta = static_cast<unsigned long>(abs(toAngle - fromAngle));
    unsigned long physicalDelta = commandDelta * SERVO3_PHYSICAL_DEGREES_PER_COMMAND_DEGREE;
    unsigned long angleBasedMs = physicalDelta * SERVO3_SETTLE_MS_PER_PHYSICAL_DEGREE;
    return max(SERVO3_SETTLE_MS, angleBasedMs);
}

void initServos() {
    servoStatePrefs.begin("sorter_servo", false);
    currentBinId = servoStatePrefs.getInt("last_bin", SERVO3_DEFAULT_BIN_ID);
    if (currentBinId < 1 || currentBinId > 6) {
        currentBinId = SERVO3_DEFAULT_BIN_ID;
    }

    pinMode(SERVO2_LED_PIN, OUTPUT);
    digitalWrite(SERVO2_LED_PIN, LOW);
    servo2.attach(SERVO2_PIN, SERVO_MIN_US, SERVO_MAX_US);
    servo3.attach(SERVO3_PIN, SERVO_MIN_US, SERVO_MAX_US);
}

void resetServos() {
    sendText("STATUS,RESET_SERVO");
    digitalWrite(SERVO2_LED_PIN, LOW);
    writeServo2(SERVO2_CLOSE_ANGLE);
    if (SERVO3_MOVE_ON_RESET) {
        currentBinId = SERVO3_DEFAULT_BIN_ID;
        writeServo3(binAngle(currentBinId));
        saveCurrentBinId();
        sendText(String("STATUS,BIN_RESET,") + String(currentBinId));
    } else {
        sendText(String("STATUS,BIN_HOLD,") + String(currentBinId));
    }
    delay(500);
}

void dropToBin(int binId) {
    if (binId < 1 || binId > 6) {
        sendText("ERROR,BAD_BIN");
        return;
    }

    sendText(String("STATUS,SORT,") + String(binId));
    if (binId != currentBinId) {
        sendText(String("STATUS,BIN_MOVE,") + String(currentBinId) + "," + String(binId));
        unsigned long waitMs = servo3MoveWaitMs(currentBinId, binId);
        writeServo3(binAngle(binId));
        sendText(String("STATUS,BIN_WAIT,") + String(waitMs));
        delay(waitMs);
        currentBinId = binId;
        saveCurrentBinId();
    } else {
        sendText(String("STATUS,BIN_KEEP,") + String(binId));
    }

    sendText("STATUS,GATE_OPEN");
    writeServo2(SERVO2_OPEN_ANGLE);
    digitalWrite(SERVO2_LED_PIN, HIGH);
    delay(DROP_HOLD_MS);

    sendText("STATUS,GATE_CLOSE");
    writeServo2(SERVO2_CLOSE_ANGLE);
    digitalWrite(SERVO2_LED_PIN, LOW);
    delay(CLOSE_SETTLE_MS);
}

void writeServo2(int angle) {
    servo2.write(constrain(angle, 0, 180));
}

void writeServo3(int angle) {
    servo3.write(constrain(angle, 0, 180));
}

void moveServo3ToBin(int binId) {
    if (binId < 1 || binId > 6) {
        sendText("ERROR,BAD_BIN");
        return;
    }
    writeServo3(binAngle(binId));
    currentBinId = binId;
    saveCurrentBinId();
    sendText(String("STATUS,BIN,") + String(binId));
}