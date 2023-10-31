import cv2
import time
import numpy as np

MODO = "MPI"

if MODO is "COCO":
    arquivoProto = "pose/body/coco/pose_deploy_linevec.prototxt"
    arquivoModelo = "pose/body/coco/pose_iter_440000.caffemodel"
    nPontos = 18
    PARES_POSES = [[1, 0], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12],
                   [12, 13], [0, 14], [0, 15], [14, 16], [15, 17]]

elif MODO is "MPI":
    arquivoProto = "pose/body/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
    arquivoModelo = "pose/body/mpi/pose_iter_160000.caffemodel"
    nPontos = 15
    PARES_POSES = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 14], [14, 8], [8, 9], [9, 10], [14, 11],
                   [11, 12], [12, 13]]

entradaLargura = 368
entradaAltura = 368
limite = 0.1

video = "imagens/body/videos/v1.mp4"
cap = cv2.VideoCapture(video)
ret, frame = cap.read()

#Salvar video de saísa, com os resultados
# 1º nome, 2º nº de quadros (fps), resolução (AxC)
gravarVideo = cv2.VideoWriter('saida.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                              (frame.shape[1], frame.shape[0]))

net = cv2.dnn.readNetFromCaffe(arquivoProto, arquivoModelo)

while cv2.waitKey(1) < 0:
    t = time.time()
    ret, frame = cap.read()
    frameCopia = np.copy(frame)
    if not ret:
        cv2.waitKey()
        break

    frameLargura = frame.shape[1]
    frameAltura = frame.shape[0]

    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (entradaLargura, entradaAltura),
                                    (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inpBlob)
    saida = net.forward()

    altura = saida.shape[2]
    largura = saida.shape[3]
    # Lista vazia para armazenar os pontos detectados
    pontos = []

    for i in range(nPontos):
        # mapa de confiança da parte do corpo correspondente
        mapaConfianca = saida[0, i, :, :]

        # encontrar a maxima global do mapaConfianca
        minVal, confianca, minLoc, point = cv2.minMaxLoc(mapaConfianca)

        # escalar o ponto para caber na imagem original
        x = (frameLargura * point[0]) / largura
        y = (frameAltura * point[1]) / altura

        if confianca > limite:
            cv2.circle(frameCopia, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frameCopia, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                        lineType=cv2.LINE_AA)

            # adicionar o ponto à lista se a probabilidade for maior que o limite
            pontos.append((int(x), int(y)))
        else:
            pontos.append(None)

    # desenhar o esqueleto ->> ligação entre os pontos
    for par in PARES_POSES:
        parteA = par[0]
        parteB = par[1]

        if pontos[parteA] and pontos[parteB]:
            cv2.line(frame, pontos[parteA], pontos[parteB], (255, 128, 0), 3, lineType=cv2.LINE_AA)
            cv2.circle(frame, pontos[parteA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(frame, pontos[parteB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

    cv2.putText(frame, "tempo levado = {:.2f} seg".format(time.time() - t), (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8,
                (255, 50, 0), 2, lineType=cv2.LINE_AA)
    # cv2.putText(imagem, "OpenPose usando OpenCV", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 50, 0), 2, lineType=cv2.LINE_AA)
    # cv2.imshow('Pontos-chave de saída', imagemCopia)
    cv2.imshow('Output-Skeleton', frame)

    gravarVideo.write(frame)

gravarVideo.release()