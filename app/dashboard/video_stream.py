import cv2
import threading
import time

latest_frame = None
lock = threading.Lock()


def update_frame(frame):
    global latest_frame

    success, buffer = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, 85]
    )

    if success:
        with lock:
            latest_frame = buffer.tobytes()


def generate_frames():

    while True:

        with lock:
            frame = latest_frame

        if frame is None:
            time.sleep(0.05)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame
            + b"\r\n"
        )

        time.sleep(0.03)