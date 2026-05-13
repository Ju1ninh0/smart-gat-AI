from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

def start_detection():

    video = cv2.VideoCapture("video/video.mp4")

    if not video.isOpened():
        print("Erro ao abrir vídeo")
        return

    while True:
        ret, frame = video.read()

        if not ret:
            print("Fim do vídeo")
            break

        results = model(frame)

        annotated_frame = results[0].plot()

        cv2.imshow("AI Detection", annotated_frame)

        if cv2.waitKey(1) == 27:
            break

    video.release()
    cv2.destroyAllWindows()