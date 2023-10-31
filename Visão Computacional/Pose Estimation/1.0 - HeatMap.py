#SLIDE 10

import cv2
import numpy as np
import matplotlib.pyplot as plt

#Para cada ponto chave, aplicamos um limite (0,1 neste caso) ao mapa de confiança.
limite=0.1

# definindo qual modelo irá usar
MODO = "MPI"

if MODO is "COCO":
    # carregando o modelo COCO baixado
    #arquitetura da rede neural
    arquivoProto = "pose/body/coco/pose_deploy_linevec.prototxt"
    #pesos do modelo
    arquivoPesos = "pose/body/coco/pose_iter_440000.caffemodel"

    # modelo COCO tem 19 pontos de reconhecimento
    nPontos = 18
    # pares de poses
    # exemplo: ligação do ponto 1 (pescoço) com o ponto 0 (Nariz), vai formar uma linha do pescoço ao nariz
    # ligação do ponto 0 (nariz) com o ponto 14 (olho direito), vai formar uma linha do nariz ao ponto do olho direito
    PARES_PONTOS = [[1, 0], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12],
                    [12, 13], [0, 14], [0, 15], [14, 16], [15, 17]]

# carregando o modelo MPII
elif MODO is "MPI":
    arquivoProto = "pose/body/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
    arquivoPesos = "pose//body/mpi/pose_iter_160000.caffemodel"

    # modelo MPII tem 15 pontos de reconheciemento
    nPontos = 15
    PARES_PONTOS = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 14], [14, 8], [8, 9], [9, 10], [14, 11],
                    [11, 12], [12, 13]]

# carregando a imagem desejada
frame = cv2.imread("imagens/body/multiple/multiple_1.jpeg")
frameCopia = np.copy(frame)
# largura da imagem
frameLargura = frame.shape[1]
# altura da imagem
frameAltura = frame.shape[0]

# Lê um modelo de rede armazenado no modelo Caffe na memória
# primeiro parâmetro é o prototexto  armazenado na varável arquivoProto, que é descrição de texto da arquitetura de rede.
# segundo parâmetro é o  caffemodel, que é o modelo da rede neural aprendida, já treinada
net = cv2.dnn.readNetFromCaffe(arquivoProto, arquivoPesos)

alturaEntrada = 368
larguraEntrada = int((alturaEntrada / frameAltura) * frameLargura)

# a imagem deve ser convertiva do formato openCV para o formato blob Caffe
# Parâmetros:
# normalizar os valores de pixels para estar entre 0,1.
#   especificar as dimensões da imagem
#       o valor médio a ser atribuido 0,0,0
#           não há necessidade de mudar os canais de cores, pois tanto o OpenCV quanto o Caffe usam o formato BGR


blobEntrada = cv2.dnn.blobFromImage(frame, 1.0 / 255, (larguraEntrada, alturaEntrada), (0, 0, 0), swapRB=False, crop=False)

# Definir o objeto preparado como o blob de entrada da rede
net.setInput(blobEntrada)

saida = net.forward()

#i = ao ponto, no exemplo abaixo o 4 corresponde ao pulso direito do modelo COCO
i = 7
mapaConfianca = saida[0, i, :, :]

# redimensionar a saída para o mesmo tamanho que a entrada
mapaConfianca = cv2.resize(mapaConfianca, (frameLargura, frameAltura))

#verificar se o mapa de confiança corresponde ao ponto chave
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
# alfa : escalar, opcional
# O valor de mistura alfa, entre 0 (transparente) e 1 (opaco)
plt.imshow(mapaConfianca, alpha=0.9)

plt.show()




