from ultralytics import YOLO
import easyocr
import cv2

model = YOLO("runs/detect/train-4/weights/best.pt")

ocr = easyocr.Reader(['en'])

def start_detection():

    video = cv2.VideoCapture("video/video3.MOV")

    if not video.isOpened():
        print("Erro ao abrir vídeo")
        return

    while True:

        ret, frame = video.read()

        if not ret:
            print("Fim do vídeo")
            break

        results = model(frame)

        for result in results:

            boxes = result.boxes

            for box in boxes:

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                plate = frame[y1:y2, x1:x2]

                if plate.size == 0:
                    continue

                plate = cv2.resize(
                    plate,
                    None,
                    fx=2,
                    fy=2
                )

                ocr_result = ocr.readtext(plate)

                text = ""

                if len(ocr_result) > 0:

                    text = ocr_result[0][1]

                    print("Placa:", text)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

        cv2.imshow(
            "AI Detection",
            frame
        )

        if cv2.waitKey(1) == 27:
            break

    video.release()

    cv2.destroyAllWindows()