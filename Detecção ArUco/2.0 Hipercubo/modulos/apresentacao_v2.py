import time
import cv2 as cv
from cv2 import aruco

from modulos import detectar_aruco


def carregar_apresentacao(video_path, id_inicial=0, list_id_aruco=[]):
    cap = cv.VideoCapture(video_path)
    count_frame = 0
    threshold = 100

    # Enquanto o vídeo estiver aberto
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Ajustar vídeo de aprensetação para FULL SCREEM
        cv.namedWindow("Video", cv.WINDOW_NORMAL)
        cv.setWindowProperty("Video", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

        # Contador de frames
        # a cada 30 frames, verifica se foi mudada a posição do cubo
        count_frame += 1
        if count_frame >= threshold:
            result = detectar_aruco.detector(
                show_detection=True, id_aruco=list_id_aruco
            )
            count_frame = 0
            if result and result[1] != id_inicial:
                break

        # time.sleep(0.03)
        cv.imshow("Video", frame)

        # Pressionar 'q' para encerrar
        key = cv.waitKey(1)
        if key == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()
