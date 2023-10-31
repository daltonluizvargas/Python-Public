import time
import cv2 as cv


def carregar_apresentacao_opencv(video_path):
    cap = cv.VideoCapture(video_path)

    # Enquanto o vídeo estiver aberto
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        time.sleep(0.03)
        cv.imshow("Video", frame)

        # Pressionar 'q' para encerrar
        key = cv.waitKey(1)
        if key == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()
