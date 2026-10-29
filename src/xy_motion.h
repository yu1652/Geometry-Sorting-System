#pragma once

#include <Arduino.h>

void initXYMotion();
void enableXYMotors(bool enabled);
bool xyHome();
bool xyMoveTo(long targetX, long targetY);
bool xyMoveToMm(float targetXmm, float targetYmm);
bool xyPushToMm(float targetXmm, float targetYmm);
void xySetCurrentPosition(long x, long y);
long xyCurrentX();
long xyCurrentY();
bool xyInSoftLimit(long x, long y);
bool xyInSoftLimitMm(float xMm, float yMm);
bool xyInTrayRangeMm(float xMm, float yMm);
long xyMmToPulseX(float xMm);
long xyMmToPulseY(float yMm);