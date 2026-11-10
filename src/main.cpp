#include <Arduino.h>

#include "comm.h"
#include <esp_system.h>
#include "config.h"
#include "pusher.h"
#include "servos.h"
#include "xy_motion.h"

enum class SystemState {
    WaitStart,
    Idle,
    WaitPush,
    Pushing,
    WaitSort,
    Sorting,
    Stopped,
};

struct PushPlan {
    float startXmm;
    float startYmm;
};

static String inputLine;
static SystemState state = SystemState::WaitStart;
static bool systemRunning = false;
static bool hostReady = false;
static uint32_t bootId = 0;
static unsigned long nextPushRetryAt = 0;

static int lastStartButtonReading = HIGH;
static int stableStartButtonState = HIGH;
static unsigned long startButtonChangedAt = 0;
static bool startButtonArmed = true;

static void initStartButton();
static void pollStartButton();
static void handleCommand(String command);
static void startSystem();
static void stopSystem();
static void resetSystem();
static void requestPushNow();
static void schedulePushRetry();
static void pollPushRetry();
static bool parsePushResponse(const String &command, PushPlan &plan);
static void handlePushPlan(const PushPlan &plan);
static bool moveToParkAfterPush();
static int parseSortBin(const String &command);
static void handleSortBin(int binId);
static const char *stateName();

void setup() {
    Serial.begin(115200);
    bootId = esp_random();
    Serial.setTimeout(30);

    initStartButton();
    initXYMotion();
    initServos();
    initPusher();
    resetServos();
    if (AUTO_HOME_ON_START) {
        pusherLiftUp();
        xyHome();
    }

    sendText("READY");
    sendText(String("BOOT,") + String(bootId));
    if (AUTO_START_SYSTEM) {
        startSystem();
    } else {
        sendText("STATUS,WAIT_START");
    }
}

void loop() {
    pollStartButton();
    pollPushRetry();

    while (Serial.available() > 0) {
        char ch = Serial.read();
        if (ch == '\n' || ch == '\r') {
            inputLine.trim();
            if (inputLine.length() > 0) {
                handleCommand(inputLine);
            }
            inputLine = "";
        } else {
            inputLine += ch;
        }
    }
}

static void initStartButton() {
    if (START_BUTTON_PIN < 0 || AUTO_START_SYSTEM) {
        return;
    }
    pinMode(START_BUTTON_PIN, START_BUTTON_USE_PULLUP ? INPUT_PULLUP : INPUT);
    int reading = digitalRead(START_BUTTON_PIN);
    lastStartButtonReading = reading;
    stableStartButtonState = reading;
    startButtonChangedAt = millis();
    startButtonArmed = stableStartButtonState != START_BUTTON_ACTIVE_LEVEL;
}

static void pollStartButton() {
    if (START_BUTTON_PIN < 0 || AUTO_START_SYSTEM) {
        return;
    }
    int reading = digitalRead(START_BUTTON_PIN);
    unsigned long now = millis();
    if (reading != lastStartButtonReading) {
        lastStartButtonReading = reading;
        startButtonChangedAt = now;
    }
    if (now - startButtonChangedAt < START_BUTTON_DEBOUNCE_MS || reading == stableStartButtonState) {
        return;
    }

    stableStartButtonState = reading;
    if (stableStartButtonState == START_BUTTON_ACTIVE_LEVEL) {
        if (startButtonArmed && !systemRunning) {
            if (!hostReady) {
                startButtonArmed = false;
                sendText("STATUS,WAIT_HOST");
                return;
            }
            sendText("STATUS,BUTTON_START");
            startSystem();
        } else if (startButtonArmed) {
            sendText("STATUS,BUTTON_IGNORED_RUNNING");
        }
        startButtonArmed = false;
    } else {
        startButtonArmed = true;
    }
}

static void handleCommand(String command) {
    command.trim();
    sendText(String("RX,") + command);

    if (command == "PING") {
        sendText("PONG");
        return;
    }
    if (command == "VERSION?") {
        sendText("VERSION,ESP32_SORTER_XY_PUSHER,0.1.0");
        return;
    }
    if (command == "STATUS?") {
        sendText(String("STATUS,MCU,") + stateName() + "," + String(xyCurrentX()) + "," + String(xyCurrentY()) + "," + String(bootId));
        return;
    }
    if (command == "HOST_READY") {
        // STOP invalidates the position assumption; RESET must rebase counters
        // after the operator has manually placed XY at the marked origin.
        if (state == SystemState::Stopped) {
            hostReady = false;
            sendText("STATUS,RESET_REQUIRED");
            return;
        }
        hostReady = true;
        sendText("STATUS,HOST_READY");
        return;
    }
    if (command == "START" && !hostReady) {
        sendText("STATUS,WAIT_HOST");
        return;
    }
    if (command == "SCAN,RETRY" && state == SystemState::WaitSort) {
        sendScanRequest();
        return;
    }
    if (command == "START") {
        startSystem();
        return;
    }
    if (command == "STOP") {
        stopSystem();
        return;
    }
    if (command == "RESET") {
        resetSystem();
        return;
    }
    if (command == "HOME") {
        pusherLiftUp();
        xyHome();
        return;
    }
    if (command.startsWith("GOTO_MM,")) {
        int comma = command.indexOf(',', 8);
        if (comma < 0) {
            sendText("ERROR,BAD_GOTO_MM");
            return;
        }
        pusherLiftUp();
        xyMoveToMm(command.substring(8, comma).toFloat(), command.substring(comma + 1).toFloat());
        return;
    }
    if (command == "PUSHER,UP") {
        pusherLiftUp();
        return;
    }
    if (command == "PUSHER,DOWN") {
        pusherLiftDown();
        return;
    }
    if (command == "SERVO2,OPEN") {
        writeServo2(SERVO2_OPEN_ANGLE);
        sendText("STATUS,SORT_GATE_OPEN");
        return;
    }
    if (command == "SERVO2,CLOSE") {
        writeServo2(SERVO2_CLOSE_ANGLE);
        sendText("STATUS,SORT_GATE_CLOSE");
        return;
    }
    if (command.startsWith("BIN,")) {
        moveServo3ToBin(command.substring(4).toInt());
        return;
    }

    if (command == "PUSH,WAIT" || command.startsWith("PUSH,NG")) {
        sendText(String("STATUS,") + command);
        schedulePushRetry();
        return;
    }

    PushPlan plan{};
    if (parsePushResponse(command, plan)) {
        handlePushPlan(plan);
        return;
    }

    int binId = parseSortBin(command);
    if (binId >= 1 && binId <= 6) {
        handleSortBin(binId);
        return;
    }
    sendText("ERROR,BAD_CMD");
}

static void startSystem() {
    if (!hostReady) {
        sendText("STATUS,WAIT_HOST");
        return;
    }
    if (systemRunning) {
        sendText("STATUS,ALREADY_RUNNING");
        return;
    }
    systemRunning = true;
    state = SystemState::Idle;
    nextPushRetryAt = 0;
    pusherLiftUp();
    sendText("STATUS,START");

    if (SERVO3_MOVE_ON_START) {
        int startBinId = SERVO3_START_BIN_ID;
        if (startBinId < 1 || startBinId > 6) {
            startBinId = SERVO3_DEFAULT_BIN_ID;
        }
        sendText(String("STATUS,BIN_START,") + String(startBinId));
        moveServo3ToBin(startBinId);
        delay(SERVO3_START_SETTLE_MS);
    }
    requestPushNow();
}

static void stopSystem() {
    hostReady = false;
    systemRunning = false;
    nextPushRetryAt = 0;
    state = SystemState::Stopped;
    pusherLiftUp();
    enableXYMotors(false);
    sendText("STATUS,STOP");
}

static void resetSystem() {
    hostReady = false;
    systemRunning = false;
    nextPushRetryAt = 0;
    xySetCurrentPosition(HOME_X_PULSE, HOME_Y_PULSE);
    enableXYMotors(true);
    resetPusher();
    resetServos();
    state = SystemState::WaitStart;
    sendText("READY");
    sendText("STATUS,WAIT_START");
}

static void requestPushNow() {
    if (!systemRunning) {
        return;
    }
    state = SystemState::WaitPush;
    nextPushRetryAt = 0;
    sendText("STATUS,REQUEST_PUSH");
    sendPushRequest();
    nextPushRetryAt = millis() + PUSH_RETRY_MS;
}

static void schedulePushRetry() {
    if (!systemRunning) {
        return;
    }
    state = SystemState::WaitPush;
    nextPushRetryAt = millis() + PUSH_RETRY_MS;
}

static void pollPushRetry() {
    if (!systemRunning || state != SystemState::WaitPush || nextPushRetryAt == 0) {
        return;
    }
    if (static_cast<long>(millis() - nextPushRetryAt) >= 0) {
        requestPushNow();
    }
}

static bool parsePushResponse(const String &command, PushPlan &plan) {
    if (!command.startsWith("PUSH,")) {
        return false;
    }
    int c1 = command.indexOf(',');
    int c2 = command.indexOf(',', c1 + 1);
    int c3 = command.indexOf(',', c2 + 1);
    if (c2 < 0 || c3 < 0) {
        return false;
    }

    String unit = command.substring(c3 + 1);
    unit.trim();
    unit.toLowerCase();
    if (unit != "mm") {
        sendText("ERROR,PUSH_UNIT_MUST_BE_MM");
        return true;
    }
    plan.startXmm = command.substring(c1 + 1, c2).toFloat();
    plan.startYmm = command.substring(c2 + 1, c3).toFloat();
    return true;
}

static void handlePushPlan(const PushPlan &plan) {
    if (!systemRunning || state != SystemState::WaitPush) {
        // Retries can queue while host vision runs; their cached responses may
        // arrive after the initial plan completed. Ignore outside WAIT_PUSH.
        sendText("STATUS,PUSH_IGNORED");
        return;
    }
    const float endXmm = PUSH_END_X_MM;
    const float endYmm = plan.startYmm;
    if (endXmm >= plan.startXmm) {
        sendText("ERROR,PUSH_DIRECTION_NOT_NEGATIVE_X");
        schedulePushRetry();
        return;
    }
    if (PUSH_MM_LIMIT_TO_TRAY_RANGE &&
        (!xyInTrayRangeMm(plan.startXmm, plan.startYmm) || !xyInTrayRangeMm(endXmm, endYmm))) {
        sendText("ERROR,PUSH_OUT_OF_TRAY");
        schedulePushRetry();
        return;
    }
    if (!xyInSoftLimitMm(plan.startXmm, plan.startYmm) || !xyInSoftLimitMm(endXmm, endYmm)) {
        sendText("ERROR,PUSH_OUT_OF_RANGE");
        schedulePushRetry();
        return;
    }

    state = SystemState::Pushing;
    nextPushRetryAt = 0;
    sendText(
        String("STATUS,PUSH_START,") + String(plan.startXmm, 2) + "," + String(plan.startYmm, 2) + "," +
        String(endXmm, 2) + "," + String(endYmm, 2)
    );

    pusherLiftUp();
    if (!xyMoveToMm(plan.startXmm, plan.startYmm)) {
        sendText("ERROR,PUSH_START_POS_FAIL");
        schedulePushRetry();
        return;
    }
    pusherLiftDown();
    if (!xyPushToMm(endXmm, endYmm)) {
        pusherLiftUp();
        sendText("ERROR,PUSH_MOVE_FAIL");
        schedulePushRetry();
        return;
    }
    delay(PUSH_END_HOLD_MS);
    pusherLiftUp();

    if (!moveToParkAfterPush()) {
        schedulePushRetry();
        return;
    }
    sendText("STATUS,PUSH_DONE");
    state = SystemState::WaitSort;
    sendText("STATUS,WAIT_SORT");
    sendScanRequest();
}

static bool moveToParkAfterPush() {
    if (!MOVE_TO_PARK_AFTER_PUSH) {
        return true;
    }
    sendText(String("STATUS,PARK_START,") + String(PARK_X_MM, 2) + "," + String(PARK_Y_MM, 2));
    if (!xyMoveToMm(PARK_X_MM, PARK_Y_MM)) {
        sendText("ERROR,PARK_POS_FAIL");
        return false;
    }
    sendText("STATUS,PARK_DONE");
    return true;
}

static int parseSortBin(const String &command) {
    if (command.startsWith("SORT,")) {
        return command.substring(5).toInt();
    }
    if (command.length() == 1 && command[0] >= '1' && command[0] <= '6') {
        return command.toInt();
    }
    return -1;
}

static void handleSortBin(int binId) {
    if (!systemRunning) {
        sendText("ERROR,NOT_RUNNING");
        return;
    }
    if (state != SystemState::WaitSort) {
        sendText("ERROR,UNEXPECTED_SORT");
        return;
    }
    state = SystemState::Sorting;
    dropToBin(binId);
    sendText("DONE");
    requestPushNow();
}

static const char *stateName() {
    switch (state) {
        case SystemState::WaitStart:
            return "WAIT_START";
        case SystemState::Idle:
            return "IDLE";
        case SystemState::WaitPush:
            return "WAIT_PUSH";
        case SystemState::Pushing:
            return "PUSHING";
        case SystemState::WaitSort:
            return "WAIT_SORT";
        case SystemState::Sorting:
            return "SORTING";
        case SystemState::Stopped:
            return "STOP";
    }
    return "UNKNOWN";
}
