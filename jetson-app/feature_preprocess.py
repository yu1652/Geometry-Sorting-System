"""Shared feature image for models trained on grayscale, contrast and edges."""

import cv2


def feature_bgr(image):
    """Pack grayscale, CLAHE and Canny features as the B, G and R channels."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    edge = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 60, 160)
    return cv2.merge([gray, clahe, edge])
