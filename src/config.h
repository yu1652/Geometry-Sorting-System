#pragma once

#include <Arduino.h>

static const bool AUTO_START_SYSTEM = false;

static const int START_BUTTON_PIN = 14;
static const int START_BUTTON_ACTIVE_LEVEL = LOW;
static const bool START_BUTTON_USE_PULLUP = true;
static const unsigned long START_BUTTON_DEBOUNCE_MS = 50;

static const int X_STEP_PIN = 25;
static const int X_DIR_PIN = 26;
static const int X_ENABLE_PIN = 27;

static const int Y_STEP_PIN = 32;
static const int Y_DIR_PIN = 33;
static const int Y_ENABLE_PIN = 23;

static const int XY_ENABLE_ACTIVE_LEVEL = LOW;

static const int X_DIR_POSITIVE_LEVEL = HIGH;
static const int Y_DIR_POSITIVE_LEVEL = LOW;

static const float X_STEPS_PER_MM = 40.0f;
static const float Y_STEPS_PER_MM = 40.0f;

static const float TRAY_WIDTH_MM = 160.0f;
static const float TRAY_HEIGHT_MM = 160.0f;

static const bool PUSH_MM_LIMIT_TO_TRAY_RANGE = false;

static const bool XY_ACCEL_ENABLE = true;

static const unsigned int XY_STEP_HIGH_US = 10;

static const unsigned int XY_STEP_LOW_US = 10;

static const unsigned int XY_START_STEP_INTERVAL_US = 2000;
static const unsigned int XY_CRUISE_STEP_INTERVAL_US = 500;

static const long XY_ACCEL_STEPS = 600;

static const unsigned int PUSH_START_STEP_INTERVAL_US = 2500;
static const unsigned int PUSH_CRUISE_STEP_INTERVAL_US = 1200;
static const long PUSH_ACCEL_STEPS = 200;

static const bool XY_HOLD_ENABLE = true;

static const long X_MIN_PULSE = 0;
static const long X_MAX_PULSE = static_cast<long>(260.0f * X_STEPS_PER_MM);
static const long Y_MIN_PULSE = 0;
static const long Y_MAX_PULSE = static_cast<long>(230.0f * Y_STEPS_PER_MM);

static const long HOME_X_PULSE = 0;
static const long HOME_Y_PULSE = 0;

static const bool USE_HOME_LIMIT_SWITCHES = false;
static const bool AUTO_HOME_ON_START = false;

static const int X_HOME_LIMIT_PIN = 16;
static const int Y_HOME_LIMIT_PIN = 17;
static const int HOME_LIMIT_ACTIVE_LEVEL = LOW;
static const bool HOME_LIMIT_USE_PULLUP = true;

static const int X_HOME_DIR_LEVEL = !X_DIR_POSITIVE_LEVEL;
static const int Y_HOME_DIR_LEVEL = !Y_DIR_POSITIVE_LEVEL;

static const float HOME_BACKOFF_MM = 5.0f;
static const float HOME_MAX_TRAVEL_MM = 600.0f;
static const unsigned int HOME_FAST_STEP_INTERVAL_US = 700;
static const unsigned int HOME_SLOW_STEP_INTERVAL_US = 1800;
static const unsigned int HOME_BACKOFF_STEP_INTERVAL_US = 1200;

static const float PUSH_END_X_MM = 40.0f;

static const bool MOVE_TO_PARK_AFTER_PUSH = true;
static const float PARK_X_MM = 10.0f;
static const float PARK_Y_MM = 10.0f;

static const unsigned long PUSH_RETRY_MS = 800;

static const int LIFT_SERVO_PIN = 18;

static const int PUSHER_LIFT_UP_ANGLE = 160;
static const int PUSHER_LIFT_DOWN_ANGLE = 115;

static const unsigned long LIFT_DOWN_STEP_MS = 25;
static const unsigned long LIFT_UP_STEP_MS = 10;
static const unsigned long LIFT_SETTLE_MS = 500;
static const unsigned long PUSH_END_HOLD_MS = 200;

static const int SERVO2_PIN = 19;
static const int SERVO3_PIN = 21;
static const int SERVO2_LED_PIN = 2;

static const int SERVO_MIN_US = 500;
static const int SERVO_MAX_US = 2500;

static const int SERVO2_CLOSE_ANGLE = 90;
static const int SERVO2_OPEN_ANGLE = 5;

static const int SERVO3_DEFAULT_BIN_ID = 3;
static const bool SERVO3_MOVE_ON_RESET = false;

static const bool SERVO3_MOVE_ON_START = true;
static const int SERVO3_START_BIN_ID = 1;
static const unsigned long SERVO3_START_SETTLE_MS = 1500;

static const int BIN_ANGLES[7] = {
    0,
    30,
    60,
    90,
    120,
    150,
    180,
};

static const unsigned long SERVO3_SETTLE_MS = 500;
static const unsigned int SERVO3_PHYSICAL_DEGREES_PER_COMMAND_DEGREE = 2;
static const unsigned long SERVO3_SETTLE_MS_PER_PHYSICAL_DEGREE = 5;
static const unsigned long DROP_HOLD_MS = 1000;
static const unsigned long CLOSE_SETTLE_MS = 1000;
