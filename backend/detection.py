import os

import cv2
import numpy as np
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "trained_model.pt")
CLASS_NAMES = ["occupied", "unoccupied-clean", "unoccupied-dirty"]
COLORS = {
    "unoccupied-clean": (0, 255, 0),
    "unoccupied-dirty": (0, 0, 255),
    "occupied": (0, 165, 255),
}

_model = None


def get_model() -> YOLO:
    global _model
    if _model is None:
        _model = YOLO(MODEL_PATH)
    return _model


def annotate_frame(frame: np.ndarray, confidence: float, draw_mode: str):
    model = get_model()
    results = model(frame, verbose=False)
    annotated = frame.copy()
    counts = {"unoccupied-clean": 0, "unoccupied-dirty": 0, "occupied": 0}

    if results and len(results) > 0:
        for box in results[0].boxes:
            conf = float(box.conf[0])
            if conf < confidence:
                continue

            cls = int(box.cls[0])
            class_name = CLASS_NAMES[cls]
            color = COLORS[class_name]
            counts[class_name] += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"{class_name} {conf:.2f}"

            if draw_mode == "circles":
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                radius = int(max(x2 - x1, y2 - y1) // 2)
                cv2.circle(annotated, (center_x, center_y), radius, color, 3)
                cv2.putText(annotated, label, (center_x - radius // 2, center_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return annotated, counts
