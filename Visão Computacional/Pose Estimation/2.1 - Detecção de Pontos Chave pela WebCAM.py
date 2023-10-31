import cv2
import time
import numpy as np

MODO = "COCO"

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

cores = [[0, 100, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255],
         [0, 255, 0], [255, 200, 100], [255, 0, 255], [0, 255, 0], [255, 200, 100], [255, 0, 255],
         [0, 0, 255], [255, 0, 0], [200, 200, 0], [255, 0, 0], [200, 200, 0], [0, 0, 0]]

# defirni dimensões para entrada do video
entradaLargura = 120
entradaAltura = 90
limite = 0.1

cap = cv2.VideoCapture(0)

net = cv2.dnn.readNetFromCaffe(arquivoProto, arquivoCaffe)

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
    fundo = np.uint8(mapaSuave > limite)

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
            cv2.circle(videoCopia, (int(x), int(y)), 8, (255, 0, 128), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(videoCopia, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                        lineType=cv2.LINE_AA)

            # Adicione o ponto à lista se a probabilidade for maior que o limite
            pontos.append((int(x), int(y)))
        else:
            pontos.append(None)

    # Desenhar esqueleto
    for pair in POSE_PARES:
        partA = pair[0]
        partB = pair[1]

        print(pontos[partB])

        if pontos[partA] and pontos[partB]:
            cv2.line(video, pontos[partA], pontos[partB], (255, 128, 0), 3, lineType=cv2.LINE_AA)
            cv2.circle(video, pontos[partA], 8, (255, 128, 0), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(video, pontos[partB], 8, (255, 128, 0), thickness=-1, lineType=cv2.FILLED)

            cv2.line(fundo, pontos[partA], pontos[partB], (255, 128, 0), 3, lineType=cv2.LINE_AA)
            cv2.circle(fundo, pontos[partA], 8, (255, 128, 0), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(fundo, pontos[partB], 8, (255, 128, 0), thickness=-1, lineType=cv2.FILLED)

        # exibindo a posição do ponto detectado
        # if pontos[partB] == (554, 400):
        #     cv2.putText(fundo, "554,400", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 50, 0), 2,
        #                 lineType=cv2.LINE_AA)
    # cv2.putText(smoothed, "time taken = {:.2f} sec".format(time.time() - tempo), (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8,(255, 50, 0), 2, lineType=cv2.LINE_AA)
    # cv2.putText(video, "OpenPose usando OpenCV", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 50, 0), 2, lineType=cv2.LINE_AA)
    cv2.imshow('Pontos', videoCopia)
    # cv2.imshow('Esqueleto', frame)
    cv2.imshow('Esqueleto', fundo)
