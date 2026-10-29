"""Lightweight smoke tests for CI: dependency imports and detection module sanity."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))


def test_core_dependencies_import():
    import cv2  # noqa: F401
    import numpy  # noqa: F401
    from ultralytics import YOLO  # noqa: F401


def test_detection_module_constants():
    from detection import CLASS_NAMES, COLORS

    assert CLASS_NAMES == ["occupied", "unoccupied-clean", "unoccupied-dirty"]
    assert set(COLORS.keys()) == set(CLASS_NAMES)


if __name__ == "__main__":
    test_core_dependencies_import()
    test_detection_module_constants()
    print("All smoke tests passed.")
