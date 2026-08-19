import cv2
import numpy as np
from pathlib import Path


INPUT = Path("video/video5.mp4")
OUTPUT = Path("video/video5_enhanced.mp4")

SCALE = 2.0


def enhance_frame(frame):
    height, width = frame.shape[:2]

    frame = cv2.resize(
        frame,
        (int(width * SCALE), int(height * SCALE)),
        interpolation=cv2.INTER_LANCZOS4
    )

    frame = cv2.fastNlMeansDenoisingColored(
        frame,
        None,
        3,
        3,
        7,
        21
    )

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    frame = cv2.cvtColor(
        cv2.merge((l, a, b)),
        cv2.COLOR_LAB2BGR
    )

    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32)

    frame = cv2.filter2D(
        frame,
        -1,
        kernel
    )

    return frame


def main():
    if not INPUT.exists():
        print(f"Vídeo não encontrado: {INPUT}")
        return

    video = cv2.VideoCapture(str(INPUT))

    if not video.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    new_width = int(width * SCALE)
    new_height = int(height * SCALE)

    codec = cv2.VideoWriter_fourcc(*"mp4v")

    output = cv2.VideoWriter(
        str(OUTPUT),
        codec,
        fps,
        (new_width, new_height)
    )

    total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    processed = 0

    print(f"Entrada: {width}x{height}")
    print(f"Saída: {new_width}x{new_height}")
    print("Processando...")

    while True:
        ret, frame = video.read()

        if not ret:
            break

        output.write(enhance_frame(frame))

        processed += 1

        if processed % 30 == 0 and total > 0:
            print(f"{(processed / total) * 100:.1f}%")

    video.release()
    output.release()

    print("Vídeo processado com sucesso.")
    print(f"Salvo em: {OUTPUT}")


if __name__ == "__main__":
    main()