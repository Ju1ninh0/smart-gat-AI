from ultralytics import YOLO

model = YOLO("runs/detect/train-2/weights/best.pt")

results = model("teste.jpg")

results[0].show()