from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/runs/detect/plate_train/weights/best.pt")

image = cv2.imread("teste.jpg")

if image is None:
    print("Erro: não foi possível abrir teste.jpg")
    exit()

results = model(
    image,
    conf=0.05,
    imgsz=640,
    verbose=False
)

total = 0

for result in results:

    for box in result.boxes:

        total += 1

        confidence = float(box.conf[0])

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        print(
            f"Placa detectada | "
            f"Confiança: {confidence:.2f}"
        )

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            image,
            f"Placa {confidence:.2f}",
            (x1, max(y1 - 10, 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

print(f"\nTotal de placas detectadas: {total}")

cv2.imshow(
    "Teste do Smart-GAT AI",
    image
)

cv2.waitKey(0)
cv2.destroyAllWindows()