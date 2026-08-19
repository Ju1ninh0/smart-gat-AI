from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/train/weights/best.pt")

video = cv2.VideoCapture("video/video5.mp4")

while True:

    ret, frame = video.read()

    if not ret:
        break

    results = model(
        frame,
        conf=0.25,
        imgsz=640,
        verbose=False
    )

    for result in results:

        for box in result.boxes:

            confidence = float(box.conf[0])

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"PLACA {confidence:.2f}",
                (x1, max(y1 - 10, 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    cv2.imshow(
        "Smart-GAT AI - Teste de Placas",
        frame
    )

    if cv2.waitKey(1) & 0xFF == 27:
        break

video.release()
cv2.destroyAllWindows()