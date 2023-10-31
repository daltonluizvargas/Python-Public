# %%
import cv2 as cv
from cv2 import aruco
import numpy as np
import base64
import ffmpeg
from random import randint

from imutils.video import VideoStream

# Variáveis de controle
frame_count_tnt = 0
frame_count_sand = 0
frame_count_gun = 0
frame_count_error = 0
detection_tnt = False
detection_sand = False
detection_gun = False
detection_error = False

# Constantes
COLOR_OVERLAY = (255, 128, 0)
COLOR_TEXT = (255, 128, 0)
COLOR_LINES = (randint(0, 255), randint(0, 255), randint(0, 255))
TEXT_SIZE = 0.75
WEBCAM_ID = 0
SAND_ID, GUN_ID, TNT_ID = 7, 2, 15

# Constantes para os caminhos dos vídeos
PATH_VIDEOS = "video/"
VIDEOS = [
    PATH_VIDEOS + "tnt.mp4",
    PATH_VIDEOS + "sand.mp4",
    PATH_VIDEOS + "gunpowder.mp4",
    PATH_VIDEOS + "error.mp4",
]

# Inicializando os vídeos
tnt = cv.VideoCapture(VIDEOS[0])
sand = cv.VideoCapture(VIDEOS[1])
gunpowder = cv.VideoCapture(VIDEOS[2])
error = cv.VideoCapture(VIDEOS[3])

# Inicializando a câmera
vs = cv.VideoCapture(WEBCAM_ID)

# Verificar se os vídeos foram abertos corretamente
if not (tnt.isOpened() and sand.isOpened() and gunpowder.isOpened() and error.isOpened() and vs.isOpened()):
    print("Erro ao abrir um ou mais vídeos.")
    tnt.release()
    sand.release()
    gunpowder.release()
    error.release()
    vs.release()
    cv.destroyAllWindows()
    exit()

# Ler os quadros dos vídeos
tnt_, tnt_video = tnt.read()
san_, sand_video = sand.read()
gun_, gunpowder_video = gunpowder.read()
error_, error_video = error.read()


marker_dict = aruco.Dictionary_get(cv.aruco.DICT_5X5_100)
param_markers = aruco.DetectorParameters_create()


def marcar_detecção(frame, ids, corners, puttext=False):
    cv.polylines(
        frame, [corners.astype(np.int32)], True, (160, 168, 172), 2, cv.LINE_AA
    )
    corners = corners.reshape(4, 2)
    corners = corners.astype(int)

    label = f"ID: {ids[0]}"

    if puttext:  # Exibir textos
        if ids[0] == 32:
            label = f"[INFO] Veiculo placa (123456) chegando..."
        if ids[0] == 15:
            label = "Prato Executivo (1)"

        cv.putText(
            frame,
            label,
            (corners[3][0], corners[0][1]),
            cv.FONT_HERSHEY_SIMPLEX,
            TEXT_SIZE,
            (255, 255, 255),
            3,
            cv.LINE_AA,
        )
        cv.putText(
            frame,
            label,
            (corners[3][0], corners[0][1]),
            cv.FONT_HERSHEY_SIMPLEX,
            TEXT_SIZE,
            COLOR_LINES,
            2,
            cv.LINE_AA,
        )


def arucoAug(bbox, id, img, imgAug):
    # Definir as coordenadas dos cantos do retângulo delimitador (bbox)
    # em torno do marcador ArUco. Cada linha atribui um par de coordenadas (x, y)
    # para cada canto.

    # superior esquerdo, superior direito
    tl = bbox[0][0][0], bbox[0][0][1]
    tr = bbox[0][1][0], bbox[0][1][1]

    # inferior direito e inferior esquerdo
    br = bbox[0][2][0], bbox[0][2][1]
    bl = bbox[0][3][0], bbox[0][3][1]

    # Extrair as dimensões da imagem de sobreposição (imgAug) em termos de altura (h),
    # largura (w) e número de canais (c).
    h, w, c = imgAug.shape

    # Criar dois conjuntos de pontos:
    # pts1 representa os cantos do retângulo do marcador ArUco no mundo real
    # pts2 representa os cantos correspondentes no plano da imagem de sobreposição.
    pts1 = np.array([tl, tr, br, bl])
    pts2 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    # Calcular a matriz de homografia que mapeia os pontos de pts2 para pts1.
    # A matriz de homografia permite mapear os pontos da imagem de sobreposição no espaço do marcador ArUco.
    matrix, _ = cv.findHomography(pts2, pts1)

    # Aplicar a transformação de perspectiva na imagem de sobreposição (imgAug)
    # usando a matriz de homografia (matrix).
    # A imagem resultante é armazenada em imgout.
    imgout = cv.warpPerspective(imgAug, matrix, (img.shape[1], img.shape[0]))

    # Preencher o polígono definido pelos pontos pts1 no retângulo delimitador do marcador
    # ArUco com a cor preta (0, 0, 0) na imagem original (img).
    # Isso é feito para remover o conteúdo da região onde o marcador será sobreposto.
    cv.fillConvexPoly(img, pts1.astype(int), (0, 0, 0))

    # Linha 1
    # Desenhar uma linha saindo da borda do marcador ArUco
    l1_inicial, l1_final = int(tl[0]) - 60, int(tl[1]) - 100
    info_dict_1 = {
        TNT_ID: "Atomic Bomb",
    }
    
    info_1 = ""
    if id in info_dict_1:
        info_1 = info_dict_1[id]

    # cv.line(imgout, tl, (l1_inicial, l1_final), (255, 255, 255), thickness=2, lineType=cv.LINE_AA)
        cv.line(
            imgout,
            tl,
            (l1_inicial, l1_final),
            COLOR_LINES,
            thickness=1,
            lineType=cv.LINE_AA,
        )

        # Exibir uma mensagem na ponta da linha
        cv.putText(
            imgout,
            info_1,
            (l1_inicial, l1_final),
            fontFace=cv.FONT_HERSHEY_SIMPLEX,
            fontScale=1.5,
            color=(255, 255, 255),
            thickness=2,
            lineType=cv.LINE_AA,
        )
        cv.putText(
            imgout,
            info_1,
            (l1_inicial, l1_final),
            fontFace=cv.FONT_HERSHEY_SIMPLEX,
            fontScale=1.5,
            color=COLOR_LINES,
            thickness=1,
            lineType=cv.LINE_AA,
        )

    # Linha 2
    l2_inicial, l2_final = int(br[0]) + 60, int(br[1]) + 100
    info_dict_2 = {
        0: "Unknown",
    }

    info_2 = ""
    if id in info_dict_2:
        info_2 = info_dict_2[id]
        
        # cv.line(imgout, br, (l2_inicial, l2_final), (255, 255, 255), thickness=2, lineType=cv.LINE_AA)
        cv.line(
            imgout,
            br,
            (l2_inicial, l2_final),
            COLOR_LINES,
            thickness=1,
            lineType=cv.LINE_AA,
        )
        cv.putText(
            imgout,
            info_2,
            (l2_inicial, l2_final),
            fontFace=cv.FONT_HERSHEY_SIMPLEX,
            fontScale=1.5,
            color=(255, 255, 255),
            thickness=2,
            lineType=cv.LINE_AA,
        )
        cv.putText(
            imgout,
            info_2,
            (l2_inicial, l2_final),
            fontFace=cv.FONT_HERSHEY_SIMPLEX,
            fontScale=1.5,
            color=COLOR_LINES,
            thickness=1,
            lineType=cv.LINE_AA,
        )


    # Adicionar a imagem de sobreposição à imagem original
    imgout = cv.addWeighted(img, 1, imgout, .7, 0)
    # imgout = img + imgout

    return imgout

# Loop infinito para manter a camera aberta
while True:
    try:
        ret, source = vs.read()

        # Converte para escala cinza
        gray_frame = cv.cvtColor(source, cv.COLOR_BGR2GRAY)

        # Guarda os resultados das detecções
        marker_corners, marker_IDs, reject = aruco.detectMarkers(
            gray_frame, marker_dict, parameters=param_markers
        )

        # Verifica o resultado da detecção e desenha o marcador
        if marker_corners:
            qtd_IDs = len(marker_IDs)
            for ids, corners in zip(marker_IDs, marker_corners):  

                if ids[0] == SAND_ID:
                    if detection_sand == False:
                        sand.set(cv.CAP_PROP_POS_FRAMES, 0)
                        frame_count_sand = 0
                        detection_sand = True
                    else:
                        if frame_count_sand == sand.get(cv.CAP_PROP_FRAME_COUNT):
                            sand.set(cv.CAP_PROP_POS_FRAMES, 0)
                            frame_count_sand = 0
                        _, sand_video = sand.read()

                    # Desenhar linhas de marcação e exibir o ID detectado
                    source = arucoAug(
                        bbox=corners, id=ids[0], img=source, imgAug=sand_video
                    )

                    marcar_detecção(frame=source, ids=ids, corners=corners)

                    frame_count_sand += 1

                if ids[0] == GUN_ID:
                    if detection_gun == False:
                        gunpowder.set(cv.CAP_PROP_POS_FRAMES, 0)
                        frame_count_gun = 0
                        detection_gun = True
                    else:
                        if frame_count_gun == gunpowder.get(cv.CAP_PROP_FRAME_COUNT):
                            gunpowder.set(cv.CAP_PROP_POS_FRAMES, 0)
                            frame_count_gun = 0
                        _, gunpowder_video = gunpowder.read()

                    # Desenhar linhas de marcação e exibir o ID detectado
                    source = arucoAug(
                        bbox=corners, id=ids[0], img=source, imgAug=gunpowder_video
                    )

                    marcar_detecção(frame=source, ids=ids, corners=corners)

                    frame_count_gun += 1

                # Ajustar função: verificar se tem 9 objetos na imagem
                if qtd_IDs > 9:
                    if ids[0] == TNT_ID:                    
                        if detection_tnt == False:
                            tnt.set(cv.CAP_PROP_POS_FRAMES, 0)
                            frame_count_tnt = 0
                            detection_tnt = True
                        else:
                            if frame_count_tnt == tnt.get(cv.CAP_PROP_FRAME_COUNT):
                                tnt.set(cv.CAP_PROP_POS_FRAMES, 0)
                                frame_count_tnt = 0
                            _, tnt_video = tnt.read()

                        # Desenhar linhas de marcação e exibir o ID detectado
                        source = arucoAug(
                            bbox=corners, id=ids[0], img=source, imgAug=tnt_video
                        )

                        marcar_detecção(frame=source, ids=ids, corners=corners)

                        frame_count_tnt += 1 

                else:
                    if ids[0] == TNT_ID:
                        if detection_error == False:
                            error.set(cv.CAP_PROP_POS_FRAMES, 0)
                            frame_count_error = 0
                            detection_error = True
                        else:
                            if frame_count_error == error.get(cv.CAP_PROP_FRAME_COUNT):
                                error.set(cv.CAP_PROP_POS_FRAMES, 0)
                                frame_count_error = 0
                            _, error_video = error.read()
                        
                        # Desenhar linhas de marcação e exibir o ID detectado
                        source = arucoAug(
                            bbox=corners, id=0, img=source, imgAug=error_video
                        )

                        marcar_detecção(frame=source, ids=ids, corners=corners)

                        frame_count_error += 1 

            cv.imshow("Resultado", source)

            # Pressionar 'q' ou 'esc' para encerrar
            if cv.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        # Liberar o vídeo da memória e encerrar todas as janelas abertas
        cv.destroyAllWindows()
        break
