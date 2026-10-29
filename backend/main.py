import os
import shutil
import tempfile
import uuid
import base64

import cv2
import imageio
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from detection import CLASS_NAMES, annotate_frame

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

app = FastAPI(title="Smart Cafe Table Detection")

app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.post("/api/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    confidence: float = Form(0.5),
    draw_mode: str = Form("circles"),
):
    data = await file.read()
    array = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not read the uploaded image.")

    annotated, counts = annotate_frame(frame, confidence, draw_mode)

    ok, buffer = cv2.imencode(".jpg", annotated)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode annotated image.")

    image_base64 = base64.b64encode(buffer).decode("utf-8")
    return {"image_base64": image_base64, "counts": counts}


def smooth_counts_timeline(timeline, window):
    """Trailing max-over-window smoothing.

    The model's per-frame detections flicker on an unchanged scene (a table
    detected as occupied in one frame can silently drop out the next), so a
    raw per-frame readout jitters even when nothing in the scene changed.
    Showing the max seen in the last `window` frames keeps the UI stable
    without hiding genuine changes, at the cost of ~window frames of lag.
    """
    smoothed = []
    for i in range(len(timeline)):
        start = max(0, i - window + 1)
        window_slice = timeline[start:i + 1]
        smoothed.append({
            cls: max(entry[cls] for entry in window_slice)
            for cls in CLASS_NAMES
        })
    return smoothed


@app.post("/api/detect/video")
async def detect_video(
    file: UploadFile = File(...),
    confidence: float = Form(0.5),
    draw_mode: str = Form("circles"),
    frame_skip: int = Form(3),
):
    suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
        shutil.copyfileobj(file.file, tmp_in)
        input_path = tmp_in.name

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        os.unlink(input_path)
        raise HTTPException(status_code=400, detail="Could not open the uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    output_name = f"{uuid.uuid4().hex}.mp4"
    output_path = os.path.join(OUTPUTS_DIR, output_name)

    writer = imageio.get_writer(output_path, fps=fps, codec="libx264", quality=7)

    counts = {"unoccupied-clean": 0, "unoccupied-dirty": 0, "occupied": 0}
    counts_timeline = []
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % max(frame_skip, 1) == 0:
                annotated, counts = annotate_frame(frame, confidence, draw_mode)
            else:
                annotated = frame

            writer.append_data(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
            counts_timeline.append(counts)
            frame_idx += 1
    finally:
        cap.release()
        writer.close()
        os.unlink(input_path)

    smoothing_window = max(1, round(fps))  # ~1 second of trailing frames
    smoothed_timeline = smooth_counts_timeline(counts_timeline, smoothing_window)

    return {
        "video_url": f"/outputs/{output_name}",
        "counts": smoothed_timeline[-1] if smoothed_timeline else counts,
        "counts_timeline": smoothed_timeline,
        "fps": fps,
        "frames_processed": frame_idx,
    }
