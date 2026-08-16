import easyocr
from ultralytics import YOLO
import cv2
import sqlite3
from datetime import datetime
from pathlib import Path
import re


MODEL_PATH = "runs/detect/train-5/weights/best.pt"
VIDEO_PATH = "video/video5.mp4"
DB_PATH = "data/smart_gat.db"

CONFIDENCE = 0.30
YOLO_SIZE = 1280

model = YOLO(MODEL_PATH)

ocr = easyocr.Reader(
    ["en"],
    gpu=True
)


def init_database():
    Path(DB_PATH).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tracking_id INTEGER,
            plate TEXT,
            confidence REAL,
            detected_at TEXT
        )
    """)

    connection.commit()
    connection.close()


def save_detection(tracking_id, plate, confidence):
    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO detections
        (tracking_id, plate, confidence, detected_at)
        VALUES (?, ?, ?, ?)
    """, (
        tracking_id,
        plate,
        confidence,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    connection.commit()
    connection.close()


def preprocess_plate(image):
    if image is None or image.size == 0:
        return None

    height, width = image.shape[:2]

    if width < 10 or height < 5:
        return None

    image = cv2.resize(
        image,
        None,
        fx=4,
        fy=4,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    gray = clahe.apply(gray)

    return cv2.cvtColor(
        gray,
        cv2.COLOR_GRAY2BGR
    )


def clean_plate(text):
    if not text:
        return None

    text = text.upper()

    text = re.sub(
        r"[^A-Z0-9]",
        "",
        text
    )

    replacements = {
        "O": "0",
        "I": "1",
        "L": "1"
    }

    text = "".join(
        replacements.get(char, char)
        for char in text
    )

    if len(text) < 4:
        return None

    return text


def read_plate(image):
    processed = preprocess_plate(image)

    if processed is None:
        return None

    try:
        results = ocr.readtext(
            processed,
            detail=1,
            paragraph=False,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )

        texts = []

        for _, text, score in results:

            if float(score) >= 0.40:
                texts.append(text)

        if not texts:
            return None

        plate = "".join(texts)

        return clean_plate(
            plate
        )

    except Exception as error:

        print(
            "Erro no OCR:",
            error
        )

        return None


def start_detection():

    init_database()

    video = cv2.VideoCapture(
        VIDEO_PATH
    )

    if not video.isOpened():

        print(
            "Erro ao abrir o vídeo:"
        )

        print(
            VIDEO_PATH
        )

        return

    detection_count = 0

    print("================================")
    print("       SMART-GAT AI")
    print("================================")
    print("Modelo:", MODEL_PATH)
    print("Vídeo:", VIDEO_PATH)
    print("Confiança:", CONFIDENCE)
    print("Resolução YOLO:", YOLO_SIZE)
    print("OCR: EasyOCR")
    print("GPU OCR: ativada")
    print("================================")

    while True:

        ret, frame = video.read()

        if not ret:

            print(
                "Fim do vídeo"
            )

            break

        results = model.track(
            frame,
            conf=CONFIDENCE,
            imgsz=YOLO_SIZE,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                confidence = float(
                    box.conf[0]
                )

                if confidence < CONFIDENCE:
                    continue

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                x1 = max(
                    0,
                    x1
                )

                y1 = max(
                    0,
                    y1
                )

                x2 = min(
                    frame.shape[1],
                    x2
                )

                y2 = min(
                    frame.shape[0],
                    y2
                )

                if x2 <= x1 or y2 <= y1:
                    continue

                if box.id is not None:

                    tracking_id = int(
                        box.id[0]
                    )

                else:

                    tracking_id = -1

                plate_crop = frame[
                    y1:y2,
                    x1:x2
                ]

                plate_text = read_plate(
                    plate_crop
                )

                detection_count += 1

                print(
                    f"[DETECÇÃO #{detection_count}] "
                    f"ID={tracking_id} "
                    f"Confiança={confidence:.3f} "
                    f"Placa={plate_text}"
                )

                save_detection(
                    tracking_id,
                    plate_text,
                    confidence
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                label = (
                    f"ID: {tracking_id} | "
                    f"{confidence:.2f}"
                )

                if plate_text:

                    label += (
                        f" | {plate_text}"
                    )

                cv2.putText(
                    frame,
                    label,
                    (
                        x1,
                        max(
                            y1 - 10,
                            30
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

        cv2.imshow(
            "Smart GAT - Tracking",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == 27:

            print(
                "Sistema encerrado."
            )

            break

    video.release()

    cv2.destroyAllWindows()

    print()
    print("================================")
    print(
        "TOTAL DE DETECÇÕES:",
        detection_count
    )
    print("================================")


if __name__ == "__main__":
    start_detection()