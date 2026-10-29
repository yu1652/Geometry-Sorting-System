#pragma once

#include <Arduino.h>

void sendText(const char *text);
void sendText(const String &text);
void sendPushRequest();
void sendScanRequest();