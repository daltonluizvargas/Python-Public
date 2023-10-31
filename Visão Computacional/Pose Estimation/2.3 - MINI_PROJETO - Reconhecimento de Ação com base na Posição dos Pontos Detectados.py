import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from Modulos import extrator_CORPO

MODO = "MPI"
salvarSaida = "N"

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

# Definir combinação de cores para não ter que ficar alterando a cada uso no código
corPonto = (14, 60, 241)
corLinha = (255, 128, 0)
corTxtPonto = (10, 216, 245)
corTxtAprov = (255, 128, 0)
corTxtWait = (14, 60, 241)

# tamanho da Font
tamFont = 1
tamLine = 2
tamCircle = 2

limite = 0.1
validaPernasJuntas = 0
validaPernasAfastadas = 0
validaBracosAbaixo = 0
validaBracosAcima = 0

# defirni dimensões para entrada do video, quanto menor mais rápido, porém perde-se a precisão
# pela webCAM 120 x 90
entradaLargura = 368
entradaAltura = 368

# imagem webCAM
# cap = cv2.VideoCapture(0)

# imagem Video
cap = cv2.VideoCapture("imagens/body/videos/v1.mp4")

if salvarSaida is "S":
    # Salvar video de saísa, com os resultados
    # 1º nome, 2º nº de quadros (fps), resolução (AxC)
    verificaVideo, videoCopia = cap.read()
    gravarVideo = cv2.VideoWriter('demonstracao.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                                  (videoCopia.shape[1], videoCopia.shape[0]))

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
        mapaConfianca = saida[0, i, :, :]
        # Encontre maxima global do mapaConfianca
        minVal, confianca, minLoc, point = cv2.minMaxLoc(mapaConfianca)
        # Escale o ponto para caber na imagem original
        x = (videoLargura * point[0]) / largura
        y = (videoAltura * point[1]) / altura

        if confianca > limite:
            cv2.circle(videoCopia, (int(x), int(y)), 4, corPonto, thickness=tamCircle, lineType=cv2.FILLED)
            cv2.putText(videoCopia, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, corTxtPonto, 0,
                        lineType=cv2.LINE_AA)
            cv2.putText(fundo, ' ' + str(int(x)) + ',' + str(int(y)), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        corTxtPonto, 0,
                        lineType=cv2.LINE_AA)

            # Adicione o ponto à lista se a probabilidade for maior que o limite
            pontos.append((int(x), int(y)))
        else:
            pontos.append((0, 0))

    # Desenhar esqueleto
    for pair in POSE_PARES:
        partA = pair[0]
        partB = pair[1]
        if pontos[partA] and pontos[partB] != (0, 0):
            cv2.line(video, pontos[partA], pontos[partB], corLinha, tamLine, lineType=cv2.LINE_AA)
            cv2.line(videoCopia, pontos[partA], pontos[partB], corLinha, tamLine, lineType=cv2.LINE_AA)

            cv2.circle(video, pontos[partA], 4, corPonto, thickness=tamCircle, lineType=cv2.FILLED)
            cv2.circle(video, pontos[partB], 4, corPonto, thickness=tamCircle, lineType=cv2.FILLED)

            cv2.line(fundo, pontos[partA], pontos[partB], corLinha, tamLine, lineType=cv2.LINE_AA)
            cv2.circle(fundo, pontos[partA], 4, corPonto, thickness=tamCircle, lineType=cv2.FILLED)
            cv2.circle(fundo, pontos[partB], 4, corPonto, thickness=tamCircle, lineType=cv2.FILLED)

    # BRAÇOS
    # também pode ser verificado um braço por vez, separando a função IF e verificando um por vez
    # por exemplo, se o braço esquerdo está acima ou se o braço direito está acima
    if extrator_CORPO.verificar_bracos_ABAIXO(pontos[0:8]) == True:
        # 50% do mov. concluído
        validaBracosAbaixo = 0.25
        # cv2.imwrite(file, videoCopia)
        cv2.line(videoCopia, pontos[0], pontos[1], corLinha, tamLine, lineType=cv2.LINE_AA)
        cv2.putText(videoCopia, "Bracos: Posicao Inicial", (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, tamFont,
                    corTxtAprov, 0,
                    lineType=cv2.LINE_AA)

    elif extrator_CORPO.verificar_bracos_ACIMA(pontos[0:8]) == True:
        # 50% do mov. concluído
        validaBracosAcima = 0.5
        # cv2.imwrite(file, videoCopia)
        cv2.line(videoCopia, pontos[0], pontos[1], corLinha, tamLine, lineType=cv2.LINE_AA)
        cv2.putText(videoCopia, "Bracos: Posicao Final", (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, tamFont,
                    corTxtAprov, 0,
                    lineType=cv2.LINE_AA)
    else:
        validaBracosAbaixo = 0
        validaBracosAcima = 0
        cv2.line(videoCopia, pontos[0], pontos[1], corLinha, tamLine, lineType=cv2.LINE_AA)
        cv2.putText(videoCopia, "Bracos: em andamento...", (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, tamFont,
                    corTxtWait, 0,
                    lineType=cv2.LINE_AA)

    # PERNAS
    if extrator_CORPO.verificar_pernas_AFASTADAS(pontos[8:14]) == True:
        # cv2.imwrite(file, videoCopia)
        # 50% do mov. concluído
        validaPernasAfastadas = 0.5
        cv2.line(videoCopia, pontos[0], pontos[1], corLinha, tamLine, lineType=cv2.LINE_AA)
        cv2.putText(videoCopia, "Pernas: Posicao Final", (50, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, tamFont, corTxtAprov,
                    0,
                    lineType=cv2.LINE_AA)
    elif extrator_CORPO.verificar_pernas_JUNTAS(pontos[8:14]) == True:
        # cv2.imwrite(file, videoCopia)
        # 50% do mov. concluído
        validaPernasJuntas = 0.25
        cv2.line(videoCopia, pontos[0], pontos[1], corLinha, tamLine, lineType=cv2.LINE_AA)
        cv2.putText(videoCopia, "Pernas: Posicao Inicial", (50, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, tamFont, corTxtAprov,
                    0,
                    lineType=cv2.LINE_AA)
    else:
        validaPernasAfastadas = 0
        validaPernasJuntas = 0
        cv2.line(videoCopia, pontos[0], pontos[1], corLinha, tamLine, lineType=cv2.LINE_AA)
        cv2.putText(videoCopia, "Pernas: em andamento...", (50, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, tamFont,
                    corTxtWait, 0,
                    lineType=cv2.LINE_AA)

    # CONTAGEM DE POLICHINELOS VÁLIDOS
    if validaBracosAcima != 0 and validaPernasAfastadas != 0:
        cv2.putText(videoCopia, 'POLICHINELOS VALIDOS: ' + str(int(validaBracosAcima + validaPernasAfastadas)), (50, 200),
                    cv2.FONT_HERSHEY_COMPLEX, tamFont,
                    corTxtAprov, 0,
                    lineType=cv2.LINE_AA)

    cv2.putText(videoCopia, "Tempo por frame: {:.2f} seg".format(time.time() - t), (50, 20),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                tamFont,
                (255, 128, 0), 0, lineType=cv2.LINE_AA)
    # cv2.putText(video, "OpenPose usando OpenCV", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 128, 0), 2, lineType=cv2.LINE_AA)

    #REDIMENSIONAR A SAÍDA
    cv2.namedWindow('N. dos Pontos + Esqueleto', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('N. dos Pontos + Esqueleto', 600, 800)

    cv2.imshow('N. dos Pontos + Esqueleto', videoCopia)
    # cv2.imshow('Pontos + Esqueleto', video)
    # cv2.imshow('Posicoes + Esqueleto', fundo)
    if salvarSaida is "S":
        gravarVideo.write(videoCopia)
if salvarSaida is "S":
    gravarVideo.release()
