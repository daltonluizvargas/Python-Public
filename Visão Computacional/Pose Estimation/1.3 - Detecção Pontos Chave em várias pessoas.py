import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

imagem = cv2.imread("imagens/body/multiple/multiple_2.jpg")

arquivoProto = "pose/body/coco/pose_deploy_linevec.prototxt"
arquivoModelo = "pose/body/coco/pose_iter_440000.caffemodel"

# Modelo COCO
nPontos = 18

MapaPontosChave = ['Nariz', 'Pescoco', 'D-Ombro', 'D-Cotov.', 'D-Pulso', 'E-Ombro', 'E-Cotov.', 'E-Pulso', 'D-Quadril', 'D-Joelho', 'D-Tornoz.',
                   'E-Qualdril', 'E-Joelho', 'E-Tornoz.', 'D-Olho', 'E-Olho', 'D-Orelha', 'E-Orelha']

#Slide 9
POSE_PARES = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
              [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
              [1, 0], [0, 14], [14, 16], [0, 15], [15, 17],
              [2, 17], [5, 16]]

# indice de PAFS correspondente ao POSE_PARES
# e.g para POSE_PARES(1,2), o PAFs estão localizados nos índices (31,32) de saída, Similarmente, (1,5) -> (39,40) e assIm por diante.
#Slide 13
mapaIndice = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44],
              [19, 20], [21, 22], [23, 24], [25, 26], [27, 28], [29, 30],
              [47, 48], [49, 50], [53, 54], [51, 52], [55, 56],
              [37, 38], [45, 46]]

cores = [[0, 100, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255],
         [0, 255, 0], [255, 200, 100], [255, 0, 255], [0, 255, 0], [255, 200, 100], [255, 0, 255],
         [0, 0, 255], [255, 0, 0], [200, 200, 0], [255, 0, 0], [200, 200, 0], [0, 0, 0]]

# Para cada ponto chave, aplicamos um limite (0,1 neste caso) ao mapa de confiança.
limite = 0.1


def getPontosChave(mapaConfianca, limite):
    # colocando bolhas nas partes detectadas
    mapaSuave = cv2.GaussianBlur(mapaConfianca, (3, 3), 0, 0)
    mapaMascara = np.uint8(mapaSuave > limite)
    pontosChaves = []

    # encontrar as bolhas (contornos)
    # Primeiro, encontre todos os contornos da região correspondentes aos pontos chave.
    # Cada contorno individual é uma matriz Numpy de coordenadas (x, y) de pontos
    contornos, _ = cv2.findContours(mapaMascara, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # para cada contorno, encontrar o máximo local
    for cnt in contornos:
        # Retorna uma nova matriz de forma e tipo dados, preenchidos com zeros
        # passando o tamanho do mascaraMapa
        mascaraBolha = np.zeros(mapaMascara.shape)

        # A função fillConvexPoly desenha um polígono convexo preenchido
        #Parâmetros:
                    # img - Imagem.
                    # pts - vértices do polígono.
                    # npts - Número de vértices do polígono.
                    # cor - cor do polígono.
                    # lineType - Tipo dos limites do polígono. Veja a descrição da linha () .
                    # shift - Número de bits fracionários nas coordenadas do vértice.
        mascaraBolha = cv2.fillConvexPoly(mascaraBolha, cnt, 1)

        # extrair o mapa de confiança para esta região, multiplicando o mapa de confiança pela mascara do mapa de confiança
        mascaraMapaConfianca = mapaSuave * mascaraBolha
        _, maxVal, _, maxLoc = cv2.minMaxLoc(mascaraMapaConfianca)
        pontosChaves.append(maxLoc + (mapaConfianca[maxLoc[1], maxLoc[0]],))

    return pontosChaves


# Encontrar conexões válidas entre as diferentes articulações de todas as pessoas presentes
#   um par válido, é uma parte do corpo unindo dois pontos chave pertencentes a mesma pessoa
#   encontrar os pares válidos seria encontrar a distância mínima entre uma articulação e todas as outras possíveis articulações
def getParesValidos(saida):
    pares_validos = []
    pares_invalidos = []
    n_interpolacao_simples = 10
    ponto_paf_th = 0.1
    conf_th = 0.7
    # loop para cada POSE_PARES
    for k in range(len(mapaIndice)):
        # A->B constitue um membro
        pafA = saida[0, mapaIndice[k][0], :, :]
        pafB = saida[0, mapaIndice[k][1], :, :]
        pafA = cv2.resize(pafA, (imagemLargura, imagemAltura))
        pafB = cv2.resize(pafB, (imagemLargura, imagemAltura))

        # Encontrar os pontos chave para o primeiro e segundo membro
        candA = pontosChave_detectados[POSE_PARES[k][0]]
        candB = pontosChave_detectados[POSE_PARES[k][1]]
        nA = len(candA)
        nB = len(candB)

        # Se os pontos chave para o par de articulacoes forem detectados
        # verifique cada articulação no candA com cada articulação em candB
        # Calcule o vetor de distância entre as duas articulacoes
        # Encontre os valores do PAF em um conjunto de pontos interpolados entre as juntas
        # Use a fórmula acima para calcular uma pontuação para marcar a conexão válida

        if (nA != 0 and nB != 0):
            par_valido = np.zeros((0, 3))
            for i in range(nA):
                max_j = -1
                pontoMaximo = -1
                encontrado = 0
                for j in range(nB):
                    # Encontrar o vetor unitário juntando os dois pontos em consideração. Isso dá a direção da linha que os une.
                    d_ij = np.subtract(candB[j][:2], candA[i][:2])
                    norm = np.linalg.norm(d_ij)
                    if norm:
                        d_ij = d_ij / norm
                    else:
                        continue
                    # Criar uma matriz de 10 pontos interpolados na linha que une os dois pontos.
                    # Encontrar p(u)
                    coordenadas_interpolacao = list(
                        zip(np.linspace(candA[i][0], candB[j][0], num=n_interpolacao_simples),
                            np.linspace(candA[i][1], candB[j][1], num=n_interpolacao_simples)))
                    # Encontrar L(p(u))
                    paf_interp = []
                    for k in range(len(coordenadas_interpolacao)):
                        paf_interp.append([pafA[int(round(coordenadas_interpolacao[k][1])), int(
                            round(coordenadas_interpolacao[k][0]))],
                                           pafB[int(round(coordenadas_interpolacao[k][1])), int(
                                               round(coordenadas_interpolacao[k][0]))]])
                    # Encontrar E
                    ponto_paf = np.dot(paf_interp, d_ij)
                    ponto_paf_media = sum(ponto_paf) / len(ponto_paf)

                    # Checar se as conexoes sao validas
                    #
                    # Se a fração de vetores interpolados alinhados com o PAF for maior que a confiança -> Par válido
                    if (len(np.where(ponto_paf > ponto_paf_th)[0]) / n_interpolacao_simples) > conf_th:
                        if ponto_paf_media > pontoMaximo:
                            max_j = j
                            pontoMaximo = ponto_paf_media
                            encontrado = 1
                # Adicionar a conexao a lista de pares validos
                if encontrado:
                    par_valido = np.append(par_valido, [[candA[i][3], candB[max_j][3], pontoMaximo]], axis=0)

            # Adiacionar as conexoes detectadas em uma lista global
            pares_validos.append(par_valido)
        else:  # Se os pontos chave não são detectados
            print("Sem conexão : k = {}".format(k))
            pares_invalidos.append(k)
            pares_validos.append([])
    return pares_validos, pares_invalidos


# Esta função cria uma lista de pontos chave pertencentes a cada pessoa
# Para cada par válido detectado, ele atribui a(s) articulação(s) a uma pessoa
def getPontosChavePessoa(pares_validos, pares_invalidos):
    # o último número em cada linha é a pontuação geral
    pessoasPontosChave = -1 * np.ones((0, 19))

    for k in range(len(mapaIndice)):
        # Primeiro, criar listas vazias para armazenar os pontos-chave de cada pessoa.
        # Em seguida, examinamos cada par, verifique se a parte A do par já está presente em qualquer uma das listas.
        # Se estiver presente, significa que o ponto chave pertence a essa lista
        # e a parte B desse par também deve pertencer a essa pessoa.
        # Assim, adicione parteB deste par à lista onde a parteA foi encontrada.
        if k not in pares_invalidos:
            partAs = pares_validos[k][:, 0]
            partBs = pares_validos[k][:, 1]
            IndiceA, indiceB = np.array(POSE_PARES[k])

            for i in range(len(pares_validos[k])):
                encontrado = 0
                pessoa_indice = -1
                for j in range(len(pessoasPontosChave)):

                    if pessoasPontosChave[j][IndiceA] == partAs[i]:
                        pessoa_indice = j
                        encontrado = 1
                        break

                if encontrado:
                    pessoasPontosChave[pessoa_indice][indiceB] = partBs[i]
                    pessoasPontosChave[pessoa_indice][-1] += lista_pontosChave[partBs[i].astype(int), 2] + \
                                                             pares_validos[k][i][2]

                # se não encontrar parteA no subconjunto, crie um novo subconjunto
                # Se parteA não estiver presente em nenhuma das listas,
                # significa que o par pertence a uma nova pessoa que não está na lista e, portanto, uma nova lista é criada.
                elif not encontrado and k < 17:
                    linha = -1 * np.ones(19)
                    linha[IndiceA] = partAs[i]
                    linha[indiceB] = partBs[i]
                    # adicione o pontosChave_escala para os dois pontos chave e o paf_escala
                    linha[-1] = sum(lista_pontosChave[pares_validos[k][i, :2].astype(int), 2]) + pares_validos[k][i][2]
                    pessoasPontosChave = np.vstack([pessoasPontosChave, linha])
    return pessoasPontosChave


imagemLargura = imagem.shape[1]
imagemAltura = imagem.shape[0]

t = time.time()
net = cv2.dnn.readNetFromCaffe(arquivoProto, arquivoModelo)

# Corrigir a altura de entrada e obter a largura de acordo com a proporção
entradaAltura = 368
entradaLargura = int((entradaAltura / imagemAltura) * imagemLargura)

inpBlob = cv2.dnn.blobFromImage(imagem, 1.0 / 255, (entradaLargura, entradaAltura),
                                (0, 0, 0), swapRB=False, crop=False)

net.setInput(inpBlob)
saida = net.forward()
print("Tempo gasto entre camadas = {}".format(time.time() - t))

pontosChave_detectados = []
lista_pontosChave = np.zeros((0, 3))
keypoint_id = 0
limite = 0.1

for part in range(nPontos):
    mapaConfianca = saida[0, part, :, :]
    mapaConfianca = cv2.resize(mapaConfianca, (imagem.shape[1], imagem.shape[0]))
    pontosChave = getPontosChave(mapaConfianca, limite)
    print("Pontos Chave - {} : {}".format(MapaPontosChave[part], pontosChave))
    keypoints_with_id = []
    for i in range(len(pontosChave)):
        keypoints_with_id.append(pontosChave[i] + (keypoint_id,))
        lista_pontosChave = np.vstack([lista_pontosChave, pontosChave[i]])
        keypoint_id += 1

    pontosChave_detectados.append(keypoints_with_id)

imagemCopia = imagem.copy()
for i in range(nPontos):
    for j in range(len(pontosChave_detectados[i])):
        cv2.circle(imagemCopia, pontosChave_detectados[i][j][0:2], 4, cores[i], -1, cv2.LINE_AA)
cv2.imshow("Pontos Chave", imagemCopia)

pares_validos, pares_invalidos = getParesValidos(saida)
pontosChavePessoa = getPontosChavePessoa(pares_validos, pares_invalidos)

for i in range(17):
    for n in range(len(pontosChavePessoa)):
        index = pontosChavePessoa[n][np.array(POSE_PARES[i])]
        if -1 in index:
            continue
        B = np.int32(lista_pontosChave[index.astype(int), 0])
        A = np.int32(lista_pontosChave[index.astype(int), 1])
        cv2.line(imagemCopia, (B[0], A[0]), (B[1], A[1]), cores[i], 2, cv2.COLOR_BGR2RGB)

plt.imshow(cv2.line(imagemCopia, (B[0], A[0]), (B[1], A[1]), cores[i], 2, cv2.COLOR_BGR5552GRAY))
plt.show()
cv2.imshow("Pose detectada", imagemCopia)
cv2.waitKey(0)
