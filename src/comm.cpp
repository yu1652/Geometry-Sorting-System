#include "comm.h"

void sendText(const char *text) {
    Serial.println(text);
}

void sendText(const String &text) {
    Serial.println(text);
}

void sendPushRequest() {
    sendText("PUSH?");
}

void sendScanRequest() {
    sendText("SCAN");
}