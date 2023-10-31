import cv2
import time
import numpy as np

# definindo qual modelo irá usar
MODELO = "MPI"

if MODELO is "COCO":
    # carregando o modelo COCO baixado
    arquivoProto = "pose/body/coco/pose_deploy_linevec.prototxt"
    arquivoModelo = "pose/body/coco/pose_iter_440000.caffemodel"

    # modelo COCO tem 19 pontos de reconhecimento
    nPontos = 18
    # pares de poses
    # exemplo: ligação do ponto 1 (pescoço) com o ponto 0 (Nariz), vai formar uma linha do pescoço ao nariz
    # ligação do ponto 0 (nariz) com o ponto 14 (olho direito), vai formar uma linha do nariz ao ponto do olho direito
    PARES_POSES = [[1, 0], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12],
                   [12, 13], [0, 14], [0, 15], [14, 16], [15, 17]]

# carregando o modelo MPII
elif MODELO is "MPI":
    arquivoProto = "pose/body/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
    arquivoModelo = "pose/body/mpi/pose_iter_160000.caffemodel"

    # modelo MPII tem 15 pontos de reconheciemento
    nPontos = 15
    PARES_POSES = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 14], [14, 8], [8, 9], [9, 10], [14, 11],
                   [11, 12], [12, 13]]

corPonto = (255, 128, 0)
corLinha = (7, 62, 248)

# carregando a imagem desejada
imagem = cv2.imread("imagens/body/single/single_1.jpg")
imagemCopia = np.copy(imagem)

# largura da imagem
imagemLargura = imagem.shape[1]
# print(imagemLargura)

# altura da imagem
imagemAltura = imagem.shape[0]
# print(imagemAltura)

# definir um limite para aplicar ao mapa de confiança, diminuindo os falsos positivos
limite = 0.1

# Lê um modelo de rede armazenado no modelo Caffe na memória
# primeiro parâmetro é o prototexto  armazenado na varável arquivoProto, que é descrição de texto da arquitetura de rede.
# segundo parâmetro é o  caffemodel, que é o modelo da rede neural aprendida, já treinada
net = cv2.dnn.readNetFromCaffe(arquivoProto, arquivoModelo)

# Retorna o tempo em segundos, será usado para verificar o tempo de execução da rede neural
tempo = time.time()
# print(tempo)
# inserindo as dimensões da imagem que entrará na rede enural
entradaLargura = 368
entradaAltura = 368

# a imagem deve ser convertiva do formato openCV para o formato blob Caffe
# Parâmetros:
# normalizar os valores de pixels para estar entre 0,1.
#   especificar as dimensões da imagem
#       o valor médio a ser atribuido 0,0,0
#           não há necessidade de mudar os canais de cores, pois tanto o OpenCV quanto o Caffe usam o formato BGR
inpBlob = cv2.dnn.blobFromImage(imagem, 1.0 / 255, (entradaLargura, entradaAltura), (0, 0, 0), swapRB=False,
                                crop=False)

# Definir o objeto preparado como o blob de entrada da rede
net.setInput(inpBlob)

# O método forward para a classe DNN no OpenCV faz um forward forward através da rede
# que é apenas outra maneira de dizer que está fazendo uma previsão
saida = net.forward()
# print(saida)

print("Tempo gasto pela rede : {:.3f}".format(time.time() - tempo))

# Lista vazia para armazenar os pontosChave detectados
pontos = []

# criando a mascara para plotar somente o esqueleto em um fundo preto
tamanho = cv2.resize(imagem, (imagemLargura, imagemAltura))

mapaSuave = cv2.GaussianBlur(tamanho, (3, 3), 0, 0)
mascaraMapa = np.uint8(mapaSuave > limite)

for i in range(nPontos):
    # mapa de confiança da parte do corpo correspondente
    mapaConfianca = saida[0, i, :, :]
    mapaConfianca = cv2.resize(mapaConfianca, (imagemLargura, imagemAltura))

    # Encontre maxima global do mapaConfianca.
    minVal, confianca, minLoc, ponto = cv2.minMaxLoc(mapaConfianca)

    if confianca > limite:
        cv2.circle(imagemCopia, (int(ponto[0]), int(ponto[1])), 4, corPonto, thickness=2, lineType=cv2.FILLED)
        # exibir a posição dos pontos
        cv2.putText(imagemCopia, (str(int(ponto[0]))) + ',' + str(int(ponto[1])), (int(ponto[0]), int(ponto[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, lineType=cv2.LINE_AA)
        # exibir os pontos
        # cv2.putText(imagemCopia, "{}".format(i), (int(ponto[0]), int(ponto[1])),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, lineType=cv2.LINE_AA)
        # print(ponto[0])
        # Adicione o ponto à lista se a probabilidade for maior que o limite
        pontos.append((int(ponto[0]), int(ponto[1])))
    else:
        pontos.append((0, 0))

for pair in PARES_POSES:
    # olhando 2 partes do corpo conectadas
    partA = pair[0]
    partB = pair[1]

    if pontos[partA] and pontos[partB] != (0, 0):
        cv2.line(imagem, pontos[partA], pontos[partB], corLinha, 1)
        cv2.circle(imagem, pontos[partA], 4, corPonto, thickness=2, lineType=cv2.LINE_AA)

        cv2.line(mascaraMapa, pontos[partA], pontos[partB], corLinha, 1)
        cv2.circle(mascaraMapa, pontos[partA], 4, corPonto, thickness=2, lineType=cv2.LINE_8)

# exibindo as saídas
cv2.imshow('PONTOS CHAVE', imagemCopia)
cv2.imshow('Esqueleto', mascaraMapa)

print("Tempo total de execução : {:.3f}".format(time.time() - tempo))

cv2.waitKey(0)
