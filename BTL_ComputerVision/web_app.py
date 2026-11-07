"""Serve the custom index.html UI and run real YOLO inference behind it."""

import math
import tempfile
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
HTML_PATH = BASE_DIR.parent / "index.html"
WEIGHTS = {
    "YOLOv8n": BASE_DIR / "weights" / "v8n_best.pt",
    "YOLOv8s": BASE_DIR / "weights" / "v8s_best.pt",
}
OUTPUT_DIR = Path(tempfile.gettempdir()) / "btl_cv_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_models = {}

app = FastAPI(title="YOLOv8 Road Safety Demo")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


def load_model(name: str) -> YOLO:
    if name not in WEIGHTS or not WEIGHTS[name].is_file():
        raise HTTPException(status_code=404, detail=f"Không tìm thấy weights cho {name}.")
    if name not in _models:
        _models[name] = YOLO(str(WEIGHTS[name]))
    return _models[name]


@app.get("/")
def home():
    if not HTML_PATH.is_file():
        raise HTTPException(status_code=404, detail="Không tìm thấy index.html ở thư mục dự án cha.")
    return FileResponse(HTML_PATH, media_type="text/html; charset=utf-8")


@app.get("/api/health")
def health():
    return {"status": "ok", "models": [name for name, path in WEIGHTS.items() if path.is_file()], "device": "CPU"}


@app.post("/api/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    model_name: str = Form("YOLOv8n"),
    conf: float = Form(0.25),
    iou: float = Form(0.45),
):
    raw = await file.read()
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Không đọc được ảnh tải lên.")
    model = load_model(model_name)
    start = time.perf_counter()
    result = model.predict(image, conf=conf, iou=iou, imgsz=640, device="cpu", verbose=False)[0]
    elapsed_ms = (time.perf_counter() - start) * 1000

    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].cpu().tolist()]
        class_id = int(box.cls[0])
        detections.append({
            "label": result.names[class_id],
            "confidence": float(box.conf[0]),
            "xyxy": [x1, y1, x2, y2],
        })

    annotated = result.plot()
    ok, encoded = cv2.imencode(".png", annotated)
    if not ok:
        raise HTTPException(status_code=500, detail="Không mã hóa được ảnh kết quả.")
    import base64

    return {
        "image": "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii"),
        "detections": detections,
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "inference_ms": elapsed_ms,
        "fps": 1000 / elapsed_ms if elapsed_ms else 0,
        "model": model_name,
    }


@app.post("/api/detect/video")
async def detect_video(
    file: UploadFile = File(...),
    model_name: str = Form("YOLOv8n"),
    conf: float = Form(0.25),
    iou: float = Form(0.45),
):
    suffix = Path(file.filename or "input.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as src:
        src.write(await file.read())
        input_path = src.name

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        Path(input_path).unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Không mở được video. Hãy thử MP4 hoặc MOV.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0

    output_name = f"result_{uuid.uuid4().hex}.webm"
    output_path = OUTPUT_DIR / output_name
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"VP80"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        Path(input_path).unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Không khởi tạo được bộ ghi WebM/VP8.")

    model = load_model(model_name)
    frame_count = 0
    started = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = model.predict(frame, conf=conf, iou=iou, imgsz=640, device="cpu", verbose=False)[0]
            writer.write(result.plot())
            frame_count += 1
    finally:
        cap.release()
        writer.release()
        Path(input_path).unlink(missing_ok=True)

    if frame_count == 0 or not output_path.is_file() or output_path.stat().st_size == 0:
        output_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Không tạo được video kết quả.")

    elapsed = time.perf_counter() - started
    return {
        "video_url": f"/outputs/{output_name}",
        "frames": frame_count,
        "processing_fps": frame_count / elapsed if elapsed else 0,
        "model": model_name,
    }
