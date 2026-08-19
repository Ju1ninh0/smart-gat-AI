from app.dashboard.video_stream import update_frame

import time
import cv2
import easyocr
import re

from pathlib import Path
from ultralytics import YOLO

from app.dashboard.database import (
    criar_banco,
    adicionar_deteccao
)


BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "runs" / "detect" / "train-5" / "weights" / "best.pt"
VIDEO_PATH = BASE_DIR / "video" / "video3.MOV"

PLATES_DIR = BASE_DIR / "data" / "plates"

CONFIDENCE = 0.30
OCR_CONFIDENCE = 0.40
YOLO_SIZE = 1280

CAMERA_NAME = "Câmera 1"

FRAMES_TO_CONFIRM = 3
PLATE_COOLDOWN = 30


model = YOLO(str(MODEL_PATH))

ocr = easyocr.Reader(
    ["en"],
    gpu=True
)


tracked_vehicles = {}
saved_vehicles = {}
last_saved_plates = {}


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

    if len(text) < 7:
        return None

    if len(text) > 8:
        return None

    old_pattern = r"^[A-Z]{3}[0-9]{4}$"
    mercosul_pattern = r"^[A-Z]{3}[0-9][A-Z][0-9]{2}$"

    if re.match(old_pattern, text):
        return text

    if re.match(mercosul_pattern, text):
        return text

    return None


def read_plate(image):

    processed = preprocess_plate(image)

    if processed is None:
        return None, 0.0

    try:

        results = ocr.readtext(
            processed,
            detail=1,
            paragraph=False,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )

        best_plate = None
        best_score = 0.0

        for _, text, score in results:

            score = float(score)

            if score < OCR_CONFIDENCE:
                continue

            plate = clean_plate(text)

            if plate and score > best_score:

                best_plate = plate
                best_score = score

        return best_plate, best_score

    except Exception as error:

        print(
            "[OCR] Erro:",
            error
        )

        return None, 0.0


def save_plate_image(
    plate,
    image,
    tracking_id
):

    if image is None or image.size == 0:
        return None

    PLATES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    safe_plate = re.sub(
        r"[^A-Z0-9]",
        "",
        plate
    )

    filename = (
        f"{safe_plate}_"
        f"ID{tracking_id}_"
        f"{cv2.getTickCount()}.jpg"
    )

    path = PLATES_DIR / filename

    cv2.imwrite(
        str(path),
        image
    )

    return str(
        path.relative_to(BASE_DIR)
    )


def register_detection(
    tracking_id,
    plate,
    ocr_confidence,
    plate_crop
):

    current_time = time.time()

    if tracking_id in saved_vehicles:
        return False

    last_time = last_saved_plates.get(plate)

    if last_time is not None:

        if current_time - last_time < PLATE_COOLDOWN:
            return False

    image_path = save_plate_image(
        plate,
        plate_crop,
        tracking_id
    )

    adicionar_deteccao(
        tracking_id=tracking_id,
        vehicle_type="Veículo",
        plate=plate,
        confidence=ocr_confidence,
        camera=CAMERA_NAME
    )

    saved_vehicles[tracking_id] = {
        "plate": plate,
        "confidence": ocr_confidence,
        "image": image_path
    }

    last_saved_plates[plate] = current_time

    print(
        f"[PLACA CONFIRMADA] "
        f"ID={tracking_id} "
        f"PLACA={plate} "
        f"OCR={ocr_confidence:.2f}"
    )

    return True


def start_detection():

    criar_banco()

    if not MODEL_PATH.exists():

        print(
            "Modelo não encontrado:"
        )

        print(
            MODEL_PATH
        )

        return

    if not VIDEO_PATH.exists():

        print(
            "Vídeo não encontrado:"
        )

        print(
            VIDEO_PATH
        )

        return

    video = cv2.VideoCapture(
        str(VIDEO_PATH)
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
    print("Confiança YOLO:", CONFIDENCE)
    print("Confiança OCR:", OCR_CONFIDENCE)
    print("Resolução:", YOLO_SIZE)
    print("OCR: EasyOCR")
    print("GPU OCR: ativada")
    print("Câmera:", CAMERA_NAME)
    print("Cooldown:", PLATE_COOLDOWN, "segundos")
    print("Dashboard: ativo")
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

                plate_text, ocr_confidence = read_plate(
                    plate_crop
                )

                if tracking_id not in tracked_vehicles:

                    tracked_vehicles[tracking_id] = {
                        "plate": None,
                        "confidence": 0.0,
                        "frames": 0,
                        "crop": None
                    }

                vehicle = tracked_vehicles[
                    tracking_id
                ]

                if plate_text:

                    vehicle["frames"] += 1

                    if (
                        ocr_confidence
                        >
                        vehicle["confidence"]
                    ):

                        vehicle["plate"] = plate_text

                        vehicle["confidence"] = (
                            ocr_confidence
                        )

                        vehicle["crop"] = (
                            plate_crop.copy()
                        )

                if (
                    vehicle["plate"]
                    and vehicle["frames"]
                    >= FRAMES_TO_CONFIRM
                ):

                    if register_detection(
                        tracking_id,
                        vehicle["plate"],
                        vehicle["confidence"],
                        vehicle["crop"]
                    ):

                        detection_count += 1

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                label = (
                    f"ID: {tracking_id} "
                    f"| YOLO: {confidence:.2f}"
                )

                if vehicle["plate"]:

                    label += (
                        f" | {vehicle['plate']}"
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

        update_frame(frame)

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
        "PLACAS CONFIRMADAS:",
        detection_count
    )
    print("================================")


if __name__ == "__main__":
    start_detection()