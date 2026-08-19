from ultralytics import YOLO


# Modelo base
model = YOLO("yolo11n.pt")


# Treinamento
model.train(
    data="dataset/data.yaml",

    epochs=100,
    imgsz=640,

    batch=8,

    patience=20,

    workers=4,

    project="runs/detect",
    name="plate_train",

    pretrained=True,

    verbose=True
)