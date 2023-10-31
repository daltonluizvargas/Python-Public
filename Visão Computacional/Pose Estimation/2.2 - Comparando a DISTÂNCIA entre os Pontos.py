import cv2
import time
import numpy as np
from Modulos import extrator_POSICAO

MODO = "MPI"

if MODO is "COCO":
    arquivoProto = "pose/body/coco/pose_deploy_linevec.prototxt"
    arquivoCaffe = "pose/body/coco/pose_iter_440000.caffemodel"
    nPontos = 18
    POSE_PARES = [[1, 0], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12],
                  [12, 13], [0, 14], [0, 15], [14, 16], [15, 17]]

elif MODO is "MPI":
    arquivoProto = "pose/body/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
    arquivoCaffe = "pose/body/mpi/pose_iter_160000.caffemodel"
    nPontos = 15
    POSE_PARES = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 14], [14, 8], [8, 9], [9, 10], [14, 11],
                  [11, 12], [12, 13]]

corPonto = (255, 128, 0)
corLinha = (7, 62, 248)

# defirni dimensões para entrada do video, quanto menor mais rápido, porém perde-se a precisão
#pela webCAM 120 x 90
entradaLargura = 368
entradaAltura = 368
limite = 0.1
tolerancia = 0.2

#imagem webCAM
# cap = cv2.VideoCapture(0)

#imagem Video
cap = cv2.VideoCapture(0)

net = cv2.dnn.readNetFromCaffe(arquivoProto, arquivoCaffe)
file = "imagenTeste.png"

while cv2.waitKey(1) < 0:
    t = time.time()
    verificaVideo, video = cap.read()

    # cópia do frame inicial
    videoCopia = np.copy(video)
    if not verificaVideo:
        cv2.waitKey()
        break

    videoLargura = video.shape[1]
    videoAltura = video.shape[0]

    # criando a mascara para plotar somente o esqueleto em um fundo preto
    tamanho = cv2.resize(video, (videoLargura, videoAltura))
    mapaSuave = cv2.GaussianBlur(tamanho, (3, 3), 0, 0)
    mascaraMapa = np.uint8(mapaSuave > limite)

    inpBlob = cv2.dnn.blobFromImage(video, 1.0 / 255, (entradaLargura, entradaAltura),
                                    (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inpBlob)
    saida = net.forward()

    altura = saida.shape[2]
    largura = saida.shape[3]
    # Lista vazia para armazenar os keypoints detectados
    pontos = []

    for i in range(nPontos):
        # mapa de confiança da parte do corpo correspondente.
        probMap = saida[0, i, :, :]
        # Encontre maxima global do mapaConfianca
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        # Escale o ponto para caber na imagem original
        x = (videoLargura * point[0]) / largura
        y = (videoAltura * point[1]) / altura

        if prob > limite:
            cv2.circle(videoCopia, (int(x), int(y)), 4, corPonto, thickness=-1, lineType=cv2.FILLED)
            cv2.putText(videoCopia, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 0,
                        lineType=cv2.LINE_AA)
            cv2.putText(mascaraMapa, ' ' + str(int(x)) + ',' + str(int(y)), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (255, 128, 0), 0,
                        lineType=cv2.LINE_AA)

            # Adicione o ponto à lista se a probabilidade for maior que o limite
            pontos.append((int(x), int(y)))
        else:
            pontos.append((0, 0))

    # Desenhar esqueleto
    for pair in POSE_PARES:
        partA = pair[0]
        partB = pair[1]
        if pontos[partA] != (0,0) and pontos[partB] != (0,0):
            cv2.line(video, pontos[partA], pontos[partB], corLinha, 1, lineType=cv2.LINE_AA)
            cv2.line(videoCopia, pontos[partA], pontos[partB], corLinha, 1, lineType=cv2.LINE_AA)

            cv2.circle(video, pontos[partA], 4, corPonto, thickness=-2, lineType=cv2.FILLED)
            cv2.circle(video, pontos[partB], 4, corPonto, thickness=-2, lineType=cv2.FILLED)

            cv2.line(mascaraMapa, pontos[partA], pontos[partB], corLinha, 1, lineType=cv2.LINE_AA)
            cv2.circle(mascaraMapa, pontos[partA], 4, corPonto, thickness=-2, lineType=cv2.FILLED)
            cv2.circle(mascaraMapa, pontos[partB], 4, corPonto, thickness=-2, lineType=cv2.FILLED)

    # BRAÇO ESQUERDO
    if extrator_POSICAO.verificar_posicao_CORPO(pontos[6:8]) == 'dobrado':
        # cv2.imwrite(file, videoCopia)
        cv2.line(mascaraMapa, pontos[0], pontos[1], (0, 128, 0), 3, lineType=cv2.LINE_AA)
        cv2.putText(mascaraMapa, "Foto Tirada", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 128, 0), 2,
                    lineType=cv2.LINE_AA)
    # print(posicoes)
    # exibindo a posição do ponto detectados
    # 512 é a posição na horizontal
    # 240 é a posição na vertical
    # if pontos[parteB] == (512,200) or pontos[parteB] == (554,200):
    #     cv2.imwrite(file, video)
    #     cv2.putText(smoothed, "Foto Tirada", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 128, 0), 2,
    #                 lineType=cv2.LINE_AA)

    # cv2.putText(smoothed, "time taken = {:.2f} sec".format(time.time() - tempo), (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8,(255, 128, 0), 2, lineType=cv2.LINE_AA)
    # cv2.putText(video, "OpenPose usando OpenCV", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 128, 0), 2, lineType=cv2.LINE_AA)
    cv2.imshow('Nº dos Pontos + Esqueleto', videoCopia)
    # cv2.imshow('Pontos + Esqueleto', video)
    cv2.imshow('Posicoes + Pontos + Esqueleto', mascaraMapa)
