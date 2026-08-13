from ultralytics import YOLO
import cv2
import sqlite3
from datetime import datetime

model = YOLO("runs/detect/train/weights/best.pt")

DB_PATH = "data/smart_gat.db"


def init_database():
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


def start_detection():

    init_database()

    video = cv2.VideoCapture("video/video3.MOV")

    if not video.isOpened():
        print("Erro ao abrir vídeo")
        return

    while True:

        ret, frame = video.read()

        if not ret:
            print("Fim do vídeo")
            break

        results = model.track(
            frame,
            conf=0.05,
            imgsz=640,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        for result in results:

            for box in result.boxes:

                if box.id is not None:
                    track_id = int(box.id[0])
                else:
                    track_id = -1

                confidence = float(box.conf[0])

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                plate = frame[y1:y2, x1:x2]

                if plate.size == 0:
                    continue

                # Aqui entra o OCR depois

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"ID: {track_id} | {confidence:.2f}",
                    (x1, max(y1 - 10, 30)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

        cv2.imshow(
            "Smart GAT - Tracking",
            frame
        )

        if cv2.waitKey(1) == 27:
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_detection()