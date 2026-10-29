#include "pusher.h"

#include <Arduino.h>
#include <ESP32Servo.h>

#include "comm.h"
#include "config.h"

static Servo liftServo;
static Servo outletGateServo;
static int currentLiftAngle = PUSHER_LIFT_UP_ANGLE;
static int currentGateAngle = OUTLET_GATE_CLOSED_ANGLE;
static bool liftPositionKnown = false;
static bool gatePositionKnown = false;

static void moveServoTo(
    Servo &servo,
    int &currentAngle,
    bool &positionKnown,
    int targetAngle,
    unsigned long stepDelayMs
) {
    int target = constrain(targetAngle, 0, 180);
    if (!positionKnown) {
        servo.write(target);
        currentAngle = target;
        positionKnown = true;
        return;
    }

    int direction = target >= currentAngle ? 1 : -1;
    while (currentAngle != target) {
        currentAngle += direction;
        servo.write(currentAngle);
        if (stepDelayMs > 0) {
            delay(stepDelayMs);
        }
    }
}

void initPusher() {
    liftServo.attach(LIFT_SERVO_PIN, SERVO_MIN_US, SERVO_MAX_US);
    outletGateServo.attach(OUTLET_GATE_SERVO_PIN, SERVO_MIN_US, SERVO_MAX_US);
    resetPusher();
    sendText("STATUS,PUSHER_INIT");
}

void resetPusher() {
    pusherLiftUp();
    outletGateClose();
}

void pusherLiftUp() {
    moveServoTo(liftServo, currentLiftAngle, liftPositionKnown, PUSHER_LIFT_UP_ANGLE, LIFT_UP_STEP_MS);
    delay(LIFT_SETTLE_MS);
    sendText("STATUS,PUSHER_UP");
}

void pusherLiftDown() {
    moveServoTo(liftServo, currentLiftAngle, liftPositionKnown, PUSHER_LIFT_DOWN_ANGLE, LIFT_DOWN_STEP_MS);
    delay(LIFT_SETTLE_MS);
    sendText("STATUS,PUSHER_DOWN");
}

void outletGateOpen() {
    moveServoTo(
        outletGateServo,
        currentGateAngle,
        gatePositionKnown,
        OUTLET_GATE_OPEN_ANGLE,
        OUTLET_GATE_STEP_MS
    );
    delay(OUTLET_GATE_SETTLE_MS);
    sendText("STATUS,OUTLET_GATE_OPEN");
}

void outletGateClose() {
    moveServoTo(
        outletGateServo,
        currentGateAngle,
        gatePositionKnown,
        OUTLET_GATE_CLOSED_ANGLE,
        OUTLET_GATE_STEP_MS
    );
    delay(OUTLET_GATE_SETTLE_MS);
    sendText("STATUS,OUTLET_GATE_CLOSED");
}