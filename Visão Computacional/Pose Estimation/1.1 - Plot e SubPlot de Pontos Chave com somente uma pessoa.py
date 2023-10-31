import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

# definindo qual modelo irá usar
MODELO = "MPI"

if MODELO is "COCO":
    # carregando o modelo COCO baixado
    arquivoProto = "pose/coco/pose_deploy_linevec.prototxt"
    arquivoModelo = "pose/coco/pose_iter_440000.caffemodel"

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
imagem = cv2.imread("imagens/body/single/single_3.jpg")
imagemCopia = np.copy(imagem)
# largura da imagem
imagemLargura = imagem.shape[1]
# print(imagemLargura)
# altura da imagem
imagemAltura = imagem.shape[0]
# print(imagemAltura)
limite = 0.1

# Lê um modelo de rede armazenado no modelo Caffe na memória
# primeiro parâmetro é o prototexto  armazenado na varável arquivoProto, que é descrição de texto da arquitetura de rede.
# segundo parâmetro é o  caffemodel, que é o modelo da rede neural aprendida, já treinada
net = cv2.dnn.readNetFromCaffe(arquivoProto, arquivoModelo)

# Retorna o tempo em segundos, será usado para verificar o tempo de execução da rede neural
t = time.time()
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

print("Tempo gasto pela rede : {:.3f}".format(time.time() - t))

# A saída é uma matriz 4D:
# A primeira dimensão é o ID da imagem (no caso de você passar mais de uma imagem para a rede).
# A segunda dimensão indica o índice de um ponto chave. O modelo produz mapas de confiança e mapas de afinidade de partes,
#  todos concatenados.
# Para o modelo COCO, ele consiste em 57 partes - 18 mapas de confiança de ponto-chave + 1 plano de fundo + 19 * 2 mapas de afinidade de parte.
# Da mesma forma, para o MPI, produz 44 pontos. Nós estaremos usando apenas os primeiros pontos que correspondem aos Keypoints.
# A terceira dimensão é a altura do mapa de saída.
# A quarta dimensão é a largura do mapa de saída.
# Verificamos se cada ponto chave está presente na imagem ou não.
# Obtemos a localização do ponto chave encontrando o máximo do mapa de confiança desse ponto chave.
# Também usamos um limite para reduzir falsas detecções.
# Uma vez que os pontos chave são detectados, apenas os plotamos na imagem.

# Forma de saída esperada da função.
altura = saida.shape[2]
largura = saida.shape[3]

# Lista vazia para armazenar os pontosChave detectados
pontos = []

for i in range(nPontos):
    # mapa de confiança da parte do corpo correspondente
    mapaConfianca = saida[0, i, :, :]

    # Encontre maxima global do mapaConfianca.
    minVal, confianca, minLoc, ponto = cv2.minMaxLoc(mapaConfianca)

    # print(confianca)

    # Escale o ponto para caber na imagem original
    x = (imagemLargura * ponto[0]) / largura
    y = (imagemAltura * ponto[1]) / altura

    if confianca > limite:
        cv2.circle(imagemCopia, (int(x), int(y)), 8, corPonto, thickness=-1, lineType=cv2.FILLED)
        cv2.putText(imagemCopia, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3,
                    lineType=cv2.LINE_AA)

        # Adicione o ponto à lista se a probabilidade for maior que o limite
        pontos.append((int(x), int(y)))
    else:
        pontos.append(None)

# desenhar o esqueleto quando temos os pontos chave apenas juntando os pares
tamanho = cv2.resize(imagem, (imagemLargura, imagemAltura))

# bolhas na região onde corresponde ao ponto chave
mapaSuave = cv2.GaussianBlur(tamanho, (3, 3), 0, 0)
mascaraMapa = np.uint8(mapaSuave > limite)

for pair in PARES_POSES:
    # olhando 2 partes do corpo conectadas
    partA = pair[0]
    partB = pair[1]

    if pontos[partA] and pontos[partB]:
        cv2.line(imagem, pontos[partA], pontos[partB], corLinha, 3)
        cv2.circle(imagem, pontos[partA], 8, corPonto, thickness=-1, lineType=cv2.LINE_AA)

        cv2.line(mascaraMapa, pontos[partA], pontos[partB], corLinha, 3)
        cv2.circle(mascaraMapa, pontos[partA], 8, corPonto, thickness=-1, lineType=cv2.LINE_AA)

# exibindo as saídas
# cv2.imshow('PONTOS CHAVE', imagemCopia)
# cv2.imshow('ESQUELETO', mascaraMapa)
fig = plt.figure()
a = fig.add_subplot(2, 2, 1)
a.set_title('Pontos')
plt.imshow(cv2.cvtColor(imagemCopia, cv2.COLOR_BGR2RGB))

a = fig.add_subplot(2, 2, 2)
a.set_title('Conexões')
plt.imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))

a = fig.add_subplot(2, 2, 3)
a.set_title('Esqueleto')
plt.imshow(cv2.cvtColor(mascaraMapa, cv2.COLOR_BGR2RGB))

plt.show()
# gravando as saídas
cv2.imwrite('resultados/Output-Keypoints.jpg', imagemCopia)
cv2.imwrite('resultados/Output-Skeleton.jpg', imagem)

print("Tempo total de execução : {:.3f}".format(time.time() - t))

cv2.waitKey(0)
