import time
import cv2
import math
import tempfile
from collections import Counter
from pathlib import Path

import gradio as gr
from ultralytics import YOLO

# Cấu hình đường dẫn model
ALL_MODELS = {
    "YOLOv8n": "weights/v8n_best.pt",
    "YOLOv8s": "weights/v8s_best.pt",
}
MODEL_PATHS = {k: v for k, v in ALL_MODELS.items() if Path(v).exists()}

VI_NAMES = {
    "person": "Người đi bộ",
    "car": "Xe hơi",
    "motorbike": "Xe máy",
    "bicycle": "Xe đạp",
}

_cache = {}

def count_objects_with_pedestrian_estimate(result):
    """Count detections and estimate pedestrians by excluding person boxes
    substantially overlapping a motorbike box. This is a heuristic, not a
    separately trained pedestrian classifier.
    """
    names = result.names
    boxes = result.boxes
    classes = [names[int(c)] for c in boxes.cls]
    xyxy = boxes.xyxy.cpu().numpy()
    motorbike_boxes = [xyxy[i] for i, name in enumerate(classes) if name == "motorbike"]
    counts = Counter(name for name in classes if name not in ("person", "motorbike"))
    counts["motorbike"] = classes.count("motorbike")

    pedestrian_count = 0
    for i, name in enumerate(classes):
        if name != "person":
            continue
        x1, y1, x2, y2 = xyxy[i]
        person_area = max(1.0, (x2 - x1) * (y2 - y1))
        overlaps_motorbike = False
        for mx1, my1, mx2, my2 in motorbike_boxes:
            iw = max(0.0, min(x2, mx2) - max(x1, mx1))
            ih = max(0.0, min(y2, my2) - max(y1, my1))
            if (iw * ih) / person_area >= 0.15:
                overlaps_motorbike = True
                break
        if not overlaps_motorbike:
            pedestrian_count += 1

    counts["pedestrian_estimate"] = pedestrian_count
    return counts

def get_model(name):
    """Tải và cache model để không phải load lại mỗi lần chạy."""
    if name not in _cache:
        m = YOLO(MODEL_PATHS[name])
        # Chạy mồi (warm-up) 1 ảnh trắng để khởi động model trên CPU
        import numpy as np
        m.predict(np.zeros((640, 640, 3), dtype="uint8"), device="cpu", verbose=False)
        _cache[name] = m
    return _cache[name]

def detect_image(image, model_name, conf):
    """Xử lý trên ảnh tĩnh."""
    if image is None:
        return None, "Chưa có ảnh. Hãy tải ảnh lên."
    if not MODEL_PATHS:
        return None, "Không tìm thấy file trong thư mục weights/."

    model = get_model(model_name)
    # Gradio truyền vào ảnh RGB, Ultralytics cần BGR
    bgr = image[:, :, ::-1]

    t0 = time.perf_counter()
    result = model.predict(bgr, conf=conf, imgsz=640, device="cpu", verbose=False)[0]
    ms = (time.perf_counter() - t0) * 1000

    # Chuyển ngược kết quả vẽ box từ BGR sang RGB để Gradio hiển thị đúng màu
    annotated = result.plot()[:, :, ::-1]

    counts = count_objects_with_pedestrian_estimate(result)
    
    lines = [f"**Model:** {model_name} | **Ngưỡng conf:** {conf:.2f}"]
    lines.append(f"**Thời gian suy luận:** {ms:.0f} ms (~{1000 / ms:.1f} FPS trên CPU)")
    lines.append("")
    
    if counts:
        lines.append("**Số lượng phát hiện:**")
        lines.append("*(Số người đi bộ là ước tính: đã loại khung person chồng đáng kể lên xe máy.)*")
        for k in ["pedestrian_estimate", "car", "motorbike", "bicycle"]:
            if k in counts:
                label = "Người đi bộ (ước tính)" if k == "pedestrian_estimate" else VI_NAMES.get(k, k)
                lines.append(f"- {label}: {counts[k]}")
        lines.append(f"- **Tổng:** {sum(counts.values())}")
    else:
        lines.append("Không phát hiện đối tượng nào. Thử hạ ngưỡng conf xuống.")
        
    return annotated, "\n".join(lines)

def detect_video(video_path, model_name, conf):
    """Xử lý trên video ngắn."""
    if video_path is None:
        return None, "Chưa có video. Hãy tải video lên."
    if not MODEL_PATHS:
        return None, "Không tìm thấy file trong thư mục weights/."

    model = get_model(model_name)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Không mở được video đầu vào. Hãy thử file MP4 hoặc MOV khác."

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if width <= 0 or height <= 0:
        cap.release()
        return None, "Không đọc được kích thước video đầu vào."
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0

    # WebM/VP8 plays in browsers. This Windows OpenCV install cannot encode
    # H.264 (avc1), and MP4V files can show NaN duration in Gradio's player.
    temp_file = tempfile.NamedTemporaryFile(prefix="yolo_result_", suffix=".webm", delete=False)
    output_path = temp_file.name
    temp_file.close()
    fourcc = cv2.VideoWriter_fourcc(*"VP80")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        cap.release()
        return None, "Không khởi tạo được bộ ghi MP4 trên máy này."
    
    frame_count = 0
    start_time = time.perf_counter()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # cv2.VideoCapture đọc ảnh BGR chuẩn cho Ultralytics, không cần đảo kênh
        results = model.predict(frame, conf=conf, imgsz=640, device="cpu", verbose=False)
        annotated_frame = results[0].plot()
        
        out.write(annotated_frame)
        frame_count += 1
        
    cap.release()
    out.release()
    if frame_count == 0 or not Path(output_path).exists() or Path(output_path).stat().st_size == 0:
        return None, "Không tạo được video kết quả; video đầu vào có thể bị lỗi hoặc codec không tương thích."
    
    total_time = time.perf_counter() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    
    info = (
        f"**Đã xử lý xong video!**\n\n"
        f"- **Model:** {model_name} | **Ngưỡng conf:** {conf:.2f}\n"
        f"- **Tổng số khung hình:** {frame_count}\n"
        f"- **Tốc độ xử lý trung bình:** ~{avg_fps:.1f} FPS (trên CPU AMD)\n"
    )
    
    return output_path, info

APP_CSS = """
.gradio-container { max-width: 1240px !important; margin: 0 auto !important; }
#hero { text-align: center; padding: 26px 16px 18px; }
#hero h1 { margin-bottom: 8px; letter-spacing: -0.03em; }
#hero p { color: var(--body-text-color-subdued); margin-top: 0; }
.panel { border: 1px solid var(--border-color-primary); border-radius: 16px !important; }
.action-row { align-items: end; }
footer { display: none !important; }
"""

with gr.Blocks(
    title="Road Safety | YOLOv8",
) as demo:
    gr.Markdown(
        "# 🚦 Phát hiện người đi bộ & phương tiện\n"
        "Chạy YOLOv8 trực tiếp trên CPU · AMD Ryzen 7 8845H",
        elem_id="hero",
    )

    with gr.Tabs():
        with gr.Tab("🖼️ Ảnh", id="image"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=1, elem_classes="panel"):
                    gr.Markdown("### 1. Tải ảnh và chọn cấu hình")
                    inp_img = gr.Image(type="numpy", label="Ảnh đường phố", height=360)
                    model_dd_img = gr.Dropdown(
                        choices=list(MODEL_PATHS.keys()),
                        value=next(iter(MODEL_PATHS), None) if MODEL_PATHS else None,
                        label="Mô hình",
                    )
                    conf_img = gr.Slider(0.1, 0.9, value=0.25, step=0.05, label="Ngưỡng tin cậy")
                    btn_img = gr.Button("Phân tích ảnh", variant="primary", size="lg")
                with gr.Column(scale=1, elem_classes="panel"):
                    gr.Markdown("### 2. Kết quả nhận diện")
                    out_img = gr.Image(label="Ảnh có khung nhận diện", height=360)
                    out_txt_img = gr.Markdown("Kết quả và thống kê sẽ hiện ở đây.")

            btn_img.click(detect_image, [inp_img, model_dd_img, conf_img], [out_img, out_txt_img])

        with gr.Tab("🎥 Video", id="video"):
            gr.Markdown("Tải video ngắn 10–20 giây. Thời gian xử lý phụ thuộc độ dài và độ phân giải video.")
            with gr.Row(equal_height=True):
                with gr.Column(scale=1, elem_classes="panel"):
                    gr.Markdown("### 1. Video đầu vào")
                    inp_vid = gr.Video(label="Chọn video", height=360)
                    model_dd_vid = gr.Dropdown(
                        choices=list(MODEL_PATHS.keys()),
                        value=next(iter(MODEL_PATHS), None) if MODEL_PATHS else None,
                        label="Mô hình",
                    )
                    conf_vid = gr.Slider(0.1, 0.9, value=0.25, step=0.05, label="Ngưỡng tin cậy")
                    btn_vid = gr.Button("Xử lý video", variant="primary", size="lg")
                with gr.Column(scale=1, elem_classes="panel"):
                    gr.Markdown("### 2. Video kết quả")
                    out_vid = gr.Video(label="Video đã nhận diện", height=360)
                    out_txt_vid = gr.Markdown("Video đã xử lý và tốc độ sẽ hiện ở đây.")

            btn_vid.click(detect_video, [inp_vid, model_dd_vid, conf_vid], [out_vid, out_txt_vid])

if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Soft(primary_hue="orange", neutral_hue="slate"),
        css=APP_CSS,
    )
