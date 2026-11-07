import time
from pathlib import Path
import numpy as np
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

# Script nằm ở thư mục gốc, còn app.py và weights/ nằm trong thư mục con.
ROOT = Path(__file__).resolve().parent
PROJECT_DIR = ROOT / "BTL_ComputerVision"
WEIGHTS_DIR = (PROJECT_DIR if PROJECT_DIR.is_dir() else ROOT) / "weights"

def benchmark_model(model_path, model_name, img_size=(640, 640), warmup=10, test_runs=100):
    print(f"\n🚀 Đang tải mô hình {model_name}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"❌ Lỗi tải mô hình {model_name}: {e}")
        return None

    # Tạo ảnh nhiễu ngẫu nhiên (dummy image) để đo tốc độ tính toán thuần túy của CPU
    dummy_img = np.random.randint(0, 255, (img_size[0], img_size[1], 3), dtype=np.uint8)

    print(f"⏳ Đang chạy khởi động (warm-up) {warmup} vòng...")
    for _ in range(warmup):
        model.predict(dummy_img, device="cpu", verbose=False)

    print(f"⏱️ Đang đo tốc độ qua {test_runs} vòng lặp...")
    start_time = time.perf_counter()
    for _ in range(test_runs):
        model.predict(dummy_img, device="cpu", verbose=False)
    end_time = time.perf_counter()

    total_time_ms = (end_time - start_time) * 1000
    avg_time_per_img = total_time_ms / test_runs
    fps = 1000 / avg_time_per_img

    print(f"✅ Hoàn thành {model_name}: {avg_time_per_img:.1f} ms/ảnh -> {fps:.1f} FPS")
    return avg_time_per_img, fps

def main():
    print("="*60)
    print("   ĐO TỐC ĐỘ SUY LUẬN YOLOV8 TRÊN CPU AMD RYZEN 7 8845H")
    print("="*60)
    
    models = {
        "YOLOv8n": WEIGHTS_DIR / "v8n_best.pt",
        "YOLOv8s": WEIGHTS_DIR / "v8s_best.pt"
    }

    results = []
    for name, path in models.items():
        res = benchmark_model(path, name)
        if res:
            results.append((name, res[0], res[1]))

    print("\n" + "="*60)
    print("🎯 KẾT QUẢ ĐÁNH GIÁ (Copy bảng này vào báo cáo)")
    print("="*60)
    print("| Model | Thời gian (ms/ảnh) | Tốc độ (FPS) | Phần cứng thực tế |")
    print("|---|---|---|---|")
    for name, ms, fps in results:
        print(f"| {name} | {ms:.1f} | ~{fps:.0f} | CPU AMD Ryzen 7 8845H |")
    print("="*60)

if __name__ == "__main__":
    main()
