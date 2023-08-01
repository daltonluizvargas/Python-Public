import cv2 as cv
from cv2 import aruco
import numpy as np
import base64
import ffmpeg
from random import randint

from imutils.video import VideoStream

rtsp_url = "rtsp://admin:Pdvr1717@192.168.20.201:554/"
http_url = "http://192.168.20.122:4747/video"


# Constantes
COLOR_OVERLAY = (255, 128, 0)
COLOR_TEXT = (255, 128, 0)
COLOR_LINES = (randint(0, 255), randint(0, 255), randint(0, 255))
TEXT_SIZE = 0.75
TIPO = "webcam"
WEBCAM_ID = 1


def conferir_veiculo(ids, id_veiculo):
    if ids[0] in id_veiculo:
        print("registrado")


def marcar_detecção(frame, ids, corners):
    cv.polylines(frame, [corners.astype(np.int32)], True, COLOR_LINES, 2, cv.LINE_AA)
    corners = corners.reshape(4, 2)
    corners = corners.astype(int)

    label = f"ID: {ids[0]}"

    if ids[0] == 32:
        label = f"[INFO] Veiculo placa (123456) chegando..."

    cv.putText(
        frame,
        label,
        (10, 50),
        cv.FONT_HERSHEY_SIMPLEX,
        TEXT_SIZE,
        (255, 255, 255),
        3,
        cv.LINE_AA,
    )
    cv.putText(
        frame,
        label,
        (10, 50),
        cv.FONT_HERSHEY_SIMPLEX,
        TEXT_SIZE,
        COLOR_TEXT,
        1,
        cv.LINE_AA,
    )


def detector(id_aruco=[], show_detection=False):
    detected = False
    # define os nomes de cada marca ArUco possível que o OpenCV suporta
    ARUCO_DICT = {
        "DICT_4X4_50": cv.aruco.DICT_4X4_50,
        "DICT_4X4_100": cv.aruco.DICT_4X4_100,
        "DICT_4X4_250": cv.aruco.DICT_4X4_250,
        "DICT_4X4_1000": cv.aruco.DICT_4X4_1000,
        "DICT_5X5_50": cv.aruco.DICT_5X5_50,
        "DICT_5X5_100": cv.aruco.DICT_5X5_100,
        "DICT_5X5_250": cv.aruco.DICT_5X5_250,
        "DICT_5X5_1000": cv.aruco.DICT_5X5_1000,
        "DICT_6X6_50": cv.aruco.DICT_6X6_50,
        "DICT_6X6_100": cv.aruco.DICT_6X6_100,
        "DICT_6X6_250": cv.aruco.DICT_6X6_250,
        "DICT_6X6_1000": cv.aruco.DICT_6X6_1000,
        "DICT_7X7_50": cv.aruco.DICT_7X7_50,
        "DICT_7X7_100": cv.aruco.DICT_7X7_100,
        "DICT_7X7_250": cv.aruco.DICT_7X7_250,
        "DICT_7X7_1000": cv.aruco.DICT_7X7_1000,
        "DICT_ARUCO_ORIGINAL": cv.aruco.DICT_ARUCO_ORIGINAL,
        "DICT_APRILTAG_16h5": cv.aruco.DICT_APRILTAG_16h5,
        "DICT_APRILTAG_25h9": cv.aruco.DICT_APRILTAG_25h9,
        "DICT_APRILTAG_36h10": cv.aruco.DICT_APRILTAG_36h10,
        "DICT_APRILTAG_36h11": cv.aruco.DICT_APRILTAG_36h11,
    }

    aruco_type = "DICT_5X5_100"
    marker_dict = aruco.Dictionary_get(ARUCO_DICT[aruco_type])
    param_markers = aruco.DetectorParameters_create()

    # Câmeras IP
    if TIPO == "rtsp":
        vs = VideoStream(rtsp_url).start()
    if TIPO == "http":
        vs = VideoStream(http_url).start()

    # Câmeras locais
    if TIPO == "webcam":
        vs = cv.VideoCapture(WEBCAM_ID)

    # Loop infinito para manter a camera aberta
    while True:
        try:
            if TIPO != "webcam":
                source = vs.read()
                if source is None:
                    continue
            else:
                ret, source = vs.read()

            # Converte para escala cinza
            gray_frame = cv.cvtColor(source, cv.COLOR_BGR2GRAY)

            # Guarda os resultados das detecções
            marker_corners, marker_IDs, reject = aruco.detectMarkers(
                gray_frame, marker_dict, parameters=param_markers
            )

            # Verifica o resultado da detecção e desenha o marcador
            if marker_corners:
                for ids, corners in zip(marker_IDs, marker_corners):
                    # Desenhar linhas de marcação e exibir o ID detected
                    if show_detection:
                        marcar_detecção(frame=source, ids=ids, corners=corners)

                    if ids[0] in id_aruco:
                        return True, ids[0]

            # Visualizar o resultado da detecção
            if show_detection:
                # Adicionar overlay
                for alpha in np.arange(0.3, 1.1, 0.9)[::-1]:
                    overlay = source.copy()
                    output = source.copy()
                    cv.rectangle(overlay, (0, 25), (525, 70), COLOR_OVERLAY, -1)
                    frame = cv.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

                cv.imshow("Resultado", source)

                # Pressionar 'q' para encerrar
                key = cv.waitKey(1)
                if key == ord("q"):
                    break

        except KeyboardInterrupt:
            # Liberar o vídeo da memória e encerrar todas as janelas abertas
            cv.destroyAllWindows()
            break
