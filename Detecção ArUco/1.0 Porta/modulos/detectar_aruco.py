import cv2 as cv
from cv2 import aruco
import numpy as np
import base64
import ffmpeg
from random import randint
import time

from modulos import porta
from modulos import speak

# Abrir a conexão com a porta
conexao_porta = porta.abrir_conexao_porta()

from imutils.video import VideoStream
rtsp_url = 'rtsp://admin:Pdvr1717@192.168.20.201:554/'
http_url = 'http://192.168.20.122:4747/video'

# Constantes
COLOR_OVERLAY = (255, 128, 0)
COLOR_TEXT    = (255, 128, 0)
COLOR_LINES   = (randint(0, 255), randint(0, 255), randint(0, 255))
TEXT_SIZE     = .75
TIPO          = 'rtsp'
WEBCAM        = 0

def verificar_deteccao(id, ids_aruco):
    resultado = False
    id_detectado = 0
    if id[0] in ids_aruco:
        resultado, id_detectado = True, id[0]
        
    return resultado, id_detectado

def verificar_posicao(x,y, id_detectado):
    id_detectado = id_detectado - 10
    id_detectado = str(id_detectado)
    print(x,y) 
    if y >= 720 and x >= 1300:
        conexao_porta.write(id_detectado.encode()) # Comando para abrir a porta         
        time.sleep(5)

def marcar_deteccao(frame, ids, corners):
    COLOR_LINES = (randint(0, 255), randint(0, 255), randint(0, 255))
    cv.polylines(
                    frame, [corners.astype(np.int32)], True, COLOR_LINES, 2, cv.LINE_AA
                )
    corners = corners.reshape(4, 2)
    (topLeft, topRight, bottomRight, bottomLeft) = corners
    topLeft = (int(topLeft[0]), int(topLeft[1]))
    topRight = (int(topRight[0]), int(topRight[1]))

    cX = int((topLeft[0] + bottomRight[0]) / 2.0)
    cY = int((topLeft[1] + bottomRight[1]) / 2.0)   

    verificar_posicao(cX, cY, ids)

    label = f'ID: {ids[0]}'

    cv.putText(frame, label, (topLeft[0], topLeft[1] - 15), cv.FONT_HERSHEY_SIMPLEX, TEXT_SIZE,(255,255,255),3,cv.LINE_AA)  
    cv.putText(frame, label, (topLeft[0], topLeft[1] - 15), cv.FONT_HERSHEY_SIMPLEX, TEXT_SIZE,COLOR_LINES,1,cv.LINE_AA)   
    cv.circle(frame, (cX, cY), 5, COLOR_LINES, -1)

    cv.imshow("Resultado", frame) 

def detector(id_aruco = [], show_detection = False, tipo = '', video_source = 1):
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
        "DICT_APRILTAG_36h11": cv.aruco.DICT_APRILTAG_36h11
    }

    aruco_type = "DICT_5X5_100"
    marker_dict = aruco.Dictionary_get(ARUCO_DICT[aruco_type])
    param_markers = aruco.DetectorParameters_create()

    if TIPO == 'rtsp':
        vs = VideoStream(rtsp_url).start() 
    if TIPO == 'webcam':
        vs = cv.VideoCapture(WEBCAM)
    if TIPO == 'http':
        vs = VideoStream(http_url).start()

    # Loop infinito para manter a camera aberta
    while True:
        try:           
            if TIPO != 'webcam':
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
                    marcar_deteccao(frame=source, ids=ids, corners=corners)
                    
                    if verificar_deteccao(id=ids, ids_aruco=id_aruco):
                        print('')                  
                                              

            # Visualizar o resultado da detecção
            if show_detection:              
                cv.imshow("Resultado", source) 
                # Pressionar 'q' para encerrar
                key = cv.waitKey(1)
                if key == ord("q"):
                    break
                
        except KeyboardInterrupt:
            # Liberar o vídeo da memória e encerrar todas as janelas abertas
            cv.destroyAllWindows()
            break    

    
