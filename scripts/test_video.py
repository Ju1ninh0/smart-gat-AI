from ultralytics import YOLO

model = YOLO("runs/detect/runs/detect/plate_train/weights/best.pt")

results = model(
    source="video/video5.mp4",
    conf=0.15,
    imgsz=1280,
    stream=True,
    save=True
)

for r in results:
    if r.boxes is not None and len(r.boxes) > 0:
        for box in r.boxes:
            conf = float(box.conf[0])
            print(f"Placa detectada | confiança: {conf:.3f}")