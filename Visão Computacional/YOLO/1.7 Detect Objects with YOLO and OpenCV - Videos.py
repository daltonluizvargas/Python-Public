import cv2
# print(cv2.__version__) #verificar a versão do OpenCV

import numpy as np
import time
import os
import matplotlib.pyplot as plt

from random import randint

MODE_TYPES = ['COCO', 'YOLO-TINY', 'OPENIMAGES']
MODE = MODE_TYPES[1]
SMALL_TEXT, AVG_TEXT = .4, .6
FONT = cv2.FONT_HERSHEY_SIMPLEX
VIDEO_SOURCE = "videos/video_pessoas01.mp4"
VIDEO_OUT = "resultado.avi"
THRESHOLD = 0.5
THRESHOLD_NMS = 0.3 # Supressão não máxima, por padrão é 0.3 (30%)
AMOSTRAS_EXIBIR = 20
AMOSTRA_ATUAL = 0
CLASSES = ['person']

if MODE == 'COCO':
    labels_path = os.path.sep.join(['darknet\data', 'coco.names'])
    LABELS = open(labels_path).read().strip().split('\n')

    config_path = os.path.sep.join(['darknet\cfg\yolov4.cfg'])
    weights_path = os.path.sep.join(['darknet\cfg\yolov4.weights'])

    net = cv2.dnn.readNet(config_path, weights_path)

if MODE == 'YOLO-TINY':
    labels_path = os.path.sep.join(['darknet\data', 'coco.names'])
    LABELS = open(labels_path).read().strip().split('\n')

    config_path = os.path.sep.join(['darknet\cfg\yolov4-tiny.cfg'])
    weights_path = os.path.sep.join(['darknet\cfg\yolov4-tiny.weights'])

    # Yolov4-Tiny é o Yolo mais leve
    net = cv2.dnn.readNet(config_path, weights_path)

if MODE == 'OPENIMAGES':
    labels_path = os.path.sep.join(['darknet\data', 'openimages.names'])
    LABELS = open(labels_path).read().strip().split('\n')

    config_path = os.path.sep.join(['darknet\cfg\yolov3-openimages.cfg'])
    weights_path = os.path.sep.join(['darknet\cfg\yolov3-openimages.weights'])

    # Carregando os pesos do Yolo OpenImages
    net = cv2.dnn.readNet('darknet\cfg\yolov3-openimages.weights', 'darknet\cfg\yolov3-openimages.cfg')

def mostrar(img):
    fig = plt.gcf()
    fig.set_size_inches(18, 10)
    plt.axis('off')
    plt.imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
    plt.show()

def redimensionar(largura, altura, largura_maxima = 600):
    if (largura > largura_maxima):
        proporcao = largura / altura
        video_largura = largura_maxima
        video_altura = int(video_largura / proporcao)
    else:
        video_largura = largura
        video_altura = video_altura
    
    return video_largura, video_altura

def blob_imagem(net, imagem, mostrar_texto = True):
    inicio = time.time()
    # Converter a imagem de entrada para o formato BLOB, ou seja, um BLOB é uma imagem pré-processada para a rede neural
    # para que a imagem se adapte ao tamanho aceito pela rede neural
    # Primeiro parâmetro: Imagem de entrada, Segundo: normalização ou escala da imagem, Terceiro: tamanho da imagem esperado pela rede neural, Quarto: Inversão de canais de cor, Quinto: Controla de parte da imagem é cortada para se encaixar no tamanho esperado
    blob = cv2.dnn.blobFromImage(imagem, 1 / 255.0, (416, 416), swapRB = True, crop = False)
    net.setInput(blob)

    # Passar a imagem para a rede neural e gerar as previsões
    layer_outputs = net.forward(ln)

    termino = time.time()
    if mostrar_texto:
        print(MODE + ' levou {:.2f} segundos '.format(termino-inicio)) 
    
    return net, imagem, layer_outputs

def deteccoes(detection, threshold, caixas, confiancas, IDclasses):
    scores = detection[5:] # Obter os valores de probalidade para cada classe
    classeID = np.argmax(scores) # Usando a função argmax para obter ID da classe com a maior probabilidade
    confianca = scores[classeID] # Porcentagem de confianca de classe

    if confianca > threshold:
        # print('scores: ', str(scores))
        # print('classe mais provavel: ', str(classeID))
        # print('confianca: ', str(confianca))

        # Redimensionar o valor das caixa de detecção relativo ao tamanho da imagem
        caixa = detection[0:4] * np.array([W, H, W, H])
        (centerX, centerY, width, height) = caixa.astype('int') # Valores das caixa delimitadoras
        x = int(centerX - (width / 2))
        y = int(centerY - (height / 2))

        caixas.append([x, y, int(width), int(height)])
        confiancas.append(float(confianca))
        IDclasses.append(classeID)
    
    return caixas, confiancas, IDclasses

# Ajustar o eixo X para não retornar valores negativos
def check_negativo(n):
    if (n < 0):
        return 0
    else:
        return n
    
def funcoes_imagem(imagem, i, confiancas, caixas, TRACKER_COLOR, LABELS, mostrar_texto = True):
    (x, y) = (caixas[i][0], caixas[i][1]) # Coordenadas das caixas
    (w, h) = (caixas[i][2], caixas[i][3]) # Tamanho das caixas

    cor = [int(c) for c in TRACKER_COLOR[IDclasses[i]]]
    cv2.rectangle(imagem, (x, y), (x + w, y + h), cor, 2)
    cv2.rectangle(imagem, (x, y), (x+150, y-20), cor, -1)
    texto = "{}: {:.4f}".format(LABELS[IDclasses[i]], confiancas[i])
    if mostrar_texto:
        print("> " + texto)
        print(x, y, w, h)
    cv2.putText(imagem, texto, (x, y - 5), FONT, SMALL_TEXT, (255,255,255), 1)

    return imagem, y, y, w, h

np.random.seed(42)
TRACKER_COLOR = np.random.randint(0, 255, size=(len(LABELS), 3), dtype='uint8')

# Lista com o nome de todas as camadas de saída(output Layer) da rede neural
ln = net.getLayerNames()
# print('Todas as camadas da rede neural: ', ln)
# print('Total de camadas: ', str(len(ln)))

# Índice das camadas de saídas são aquelas que não estão ligadas com nenhuma outra saída, por isso a chamda da função getUnconnectedOutLayers
# print('Camadas de saída: ', net.getUnconnectedOutLayers())

# Associando os IDs das camadas de saídas com os nomes
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# print(ln)

# Carregar o video de entrada
cap = cv2.VideoCapture(VIDEO_SOURCE)
ok, video = cap.read()
if not ok:
    print('Erro ao carregar o video. Parando...')

video_largura = video.shape[1]
video_altura = video.shape[0]
video_largura, video_altura = redimensionar(video.shape[1], video.shape[0])

fourcc = cv2.VideoWriter_fourcc(*"XVID")
writer_VIDEO = cv2.VideoWriter(VIDEO_OUT, fourcc, 25, (video_largura, video_altura))

while True:
    ok, frame = cap.read()
    if not ok:
        print('Erro ao carregar o video. Parando...')
        break
    
    t = time.time()
    frame = cv2.resize(frame, (video_largura, video_altura))

    try:
        (H, W) = frame.shape[:2]
    except:
        print('Erro!')
        continue
    
    frame_cp = frame.copy()
    net, frame, layer_outputs = blob_imagem(net, frame)

    caixas = []
    confiancas = []
    IDclasses = []

    for output in layer_outputs: # Percorrendo cada camada de saída
        for detection in output: # Percorrendo cada uma das detecções
            caixas, confiancas, IDclasses = deteccoes(detection, THRESHOLD, caixas, confiancas, IDclasses)
    
    objs = cv2.dnn.NMSBoxes(caixas, confiancas, THRESHOLD, THRESHOLD_NMS)

    # Se o npumero de objetos for maior do que 0
    if len(objs) > 0:
        for i in objs.flatten(): # Flatten transforma de uma matriz para o formato de vetor
            # if LABELS[IDclasses[i]] in CLASSES:
            frame, x, y, w, h = funcoes_imagem(frame, i, confiancas, caixas, TRACKER_COLOR, LABELS, mostrar_texto = False)
            objeto = frame_cp[y:y + h, x:x + w]

    cv2.putText(frame, ' frame processado em {:.2f} segundos'.format(time.time() - t), (20, video_altura - 20), FONT, SMALL_TEXT, (250, 250, 250), 0, lineType=cv2.LINE_AA)

    if AMOSTRA_ATUAL <= AMOSTRAS_EXIBIR:
        cv2.imshow('Atual', frame)
        AMOSTRA_ATUAL += 1
    
    writer_VIDEO.write(frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
            break

writer_VIDEO.release()
cv2.destroyAllWindows()