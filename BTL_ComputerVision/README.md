# BTL Computer Vision — YOLOv8 Road Safety

## Chạy giao diện tùy chỉnh (`index.html`)

Sau khi clone repo, cài thư viện (nên dùng Python 3.10 hoặc 3.11):

```powershell
python -m pip install -r requirements.txt
```

Đảm bảo hai file `weights/v8n_best.pt` và `weights/v8s_best.pt` đã có. Mở PowerShell tại thư mục `BTL_ComputerVision` (thư mục chứa `web_app.py` và `weights/`), sau đó chạy:

```powershell
python -m uvicorn web_app:app --host 127.0.0.1 --port 7860
```

Mở http://127.0.0.1:7860. Server sẽ phục vụ `index.html` ở thư mục cha và gọi model YOLO thật trên CPU. Không mở `index.html` bằng cách double-click; khi đó trang không kết nối được Python backend.

## Chạy giao diện Gradio cũ

```powershell
python app.py
```

Sau đó mở http://127.0.0.1:7860.

## Cấu trúc

```text
BTL_ComputerVision/
├── index.html                 # Giao diện web tùy chỉnh
└── BTL_ComputerVision/
    ├── app.py                 # Giao diện Gradio
    ├── web_app.py             # API suy luận + phục vụ index.html
    └── weights/
        ├── v8n_best.pt
        └── v8s_best.pt
```
