#%%
import seaborn as sns

sns.set()

import csv
import glob
import os
import random

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
# Comece por instalar TensorFlow I/O, o que tornará mais fácil para você para carregar arquivos de áudio off disco.
# pip install tensorflow_io
import tensorflow_io as tfio
# A biblioteca TFLite Model Maker simplifica o processo de adaptação e conversão de um modelo de rede neural TensorFlow para dados de entrada específicos ao implantar este modelo para aplicativos de ML no dispositivo.
# pip install tflite-model-maker
import tflite_model_maker as mm
from IPython.display import Audio, Image, display
from tensorflow.keras.layers import Activation, Dense, Input
# Definição do modelo, camadas, função de ativação,...
from tensorflow.keras.models import Sequential
from tflite_model_maker import audio_classifier
# tqdm é uma biblioteca em Python usada para criar medidores de progresso ou barras de progresso. O nome tqdm vem do nome árabe taqaddum, que significa 'progresso'
from tqdm import tqdm

#%%
# %%
# Relembrando sobre o YAMNet:

# YAMNet é uma rede neural pré-treinada que emprega a arquitetura MobileNetV1. Ele pode usar uma forma de onda de áudio como entrada e fazer previsões independentes para cada um dos 521 eventos de áudio do modelo.

# Internamente, o modelo extrai "quadros" do sinal de áudio e processa lotes desses quadros. Esta versão do modelo usa quadros com 0,96 segundo de duração e extrai um quadro a cada 0,48 segundos.

# O modelo aceita uma matriz 1-D float32(Tensor ou NumPy) contendo uma forma de onda de comprimento arbitrário, como representado de canal único (mono) com amostras de 16 kHz no intervalo [-1.0, +1.0]. 

# O modelo retorna 3 saídas, incluindo as notas de classe, embeddings (que você vai usar para a aprendizagem de transferência), e o log mel espectrograma. 

# Um uso específico do YAMNet é como um extrator de recursos de alto nível alimentando apenas uma comada oculta densa (tf.keras.layers.Dense) com a dimensão de 1024 neurônios. Ou seja,iremos usar recursos de entrada do modelo de base (YAMNet) e alimentá-los de forma mais rasa em nosso modelo. Em outras palavras, treinamos a rede em uma pequena quantidade de dados para a classificação de áudio sem a necessidade de uma grande quantidade de dados rotulados.

# %%
# Carregando o modelo que prevê 521 eventos de áudio
# https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/yamnet_class_map.csv
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

#%%
# O arquivo de com o nome das classes será carregado a partir dos modelos ativos e está presente em model.class_map_path() . O carregamento será na vaiável class_names.
def class_names_from_csv(class_map_csv_text):
    '''Retorna a lista de nomes de classes correspondentes'''
    class_names = []
    with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # display_name é o nome da coluna com o nome das classes
            class_names.append(row['display_name'])
    
    return class_names

class_map_path = yamnet_model.class_map_path().numpy()
class_names = class_names_from_csv(class_map_path)

# %%
class_names

# %%
# Exibir a quantidade de classes no modelo
len(class_names)
# %%
# Exibir as 20 primeiras classes dos modelo
for name in class_names[:20]:
    print(name)
print('...')

#%%
# Neste exemplo usares o conjunto de dados Birds, que é uma coleção educacional com 5 tipos de cantos de pássaros:
# White-breasted Wood-Wren (uirapuru-de-peito-branco)
# House Sparrow (Pardal)
# Red Crossbill (Cruza-bico)
# Chestnut-crowned Antpitta (Grallaria ruficapilla)
# Azara's Spinetail (espineta de Azara)
# O áudio original veio do Xeno-canto, que é um site dedicado a compartilhar sons de pássaros de todo o mundo.

# Os áudios já estão divididos em pastas de teste e treinamento. Dentro de cada pasta, há uma pasta para cada pássaro, usando seu bird_code como nome.

# Os áudios são todos mono e com taxa de amostragem de 16kHz.

# Para obter mais informações sobre cada arquivo, você pode ler o arquivo metadata.csv. Ele contém todos os autores dos arquivos, lincensas e mais algumas informações. Você não precisará ler neste tutorial.
DATASET_PATH = 'Datasets/small_birds_dataset/'
DATA = 'Datasets/small_birds_dataset/metadata.csv'

#%%
metadata = pd.read_csv(DATA)
metadata.head()

#%%
bird_code_to_name = {
  'wbwwre1': 'White-breasted Wood-Wren',
  'houspa': 'House Sparrow',
  'redcro': 'Red Crossbill',  
  'chcant2': 'Chestnut-crowned Antpitta',
  'azaspi1': "Azara's Spinetail",   
}

birds_images = {
  'wbwwre1': DATASET_PATH + '\images\White-breasted Wood-Wren.jpg', # 	Alejandro Bayer Tamayo from Armenia, Colombia 
  'houspa': DATASET_PATH + '\images\House Sparrow.jpg', # 	Diliff
  'redcro': DATASET_PATH + '\images\Red Crossbill.jpg', #  Elaine R. Wilson, www.naturespicsonline.com
  'chcant2': DATASET_PATH + '\images\Chestnut-crowned Antpitta.jpg', # 	Mike's Birds from Riverside, CA, US
  'azaspi1': DATASET_PATH + "\images\Azara's Spinetail.jpg", # https://www.inaturalist.org/photos/76608368
}

#%%
test_files = os.path.join(DATASET_PATH, "test/*/*.wav")

#%%
# Função para selecionar arquivos de áudio aleatórios da base de dados
def get_random_audio_file():
  test_list = glob.glob(test_files)
  random_audio_path = random.choice(test_list)
  return random_audio_path

# Função para carregar arquivos de áudio e a imagem correspondente baseando-se no path do arquivo (contendo o código/id do pássaro)
def get_bird_data(audio_path, plot = True):
    wav_data, sample_rate = librosa.load(audio_path, sr=16000)   

    if plot:
        bird_code = audio_path.split('\\')[-2]
        print(f'Nome do pássaro: {bird_code_to_name[bird_code]}')
        print(f'Código do pássaro: {bird_code}')
        display(Image(birds_images[bird_code]))

        plttitle = f'{bird_code_to_name[bird_code]} ({bird_code})'
        plt.title(plttitle)
        plt.plot(wav_data)
        display(Audio(wav_data, rate=sample_rate))

    return wav_data

print('[INFO] funções e estruturas de dados criadas')

#%%
random_audio = get_random_audio_file()
get_bird_data(random_audio)

#%%
# Os resultados do modelo são:
  # scores: previsões - pontuações para cada uma das 521 classes;
  # embeddings - recursos YAMNet;
  # spectrogram - espectrogramas
random_audio = get_random_audio_file()
scores, embeddings, spectrogram = yamnet_model(get_bird_data(random_audio))
class_scores = tf.reduce_mean(scores, axis=0)
top_class = tf.argmax(class_scores)
infered_class = class_names[top_class]

print(f'[INFO] o som principal é: {infered_class}')
print(f'[INF] a forma dos embeddings: {embeddings.shape}')

#%%
# Extrair as emdeddings com o modelo YAMNet
# random_audio = get_random_audio_file()

def extract_embedding(wav_data):    
    scores, embeddings, spectrogram = yamnet_model(get_bird_data(wav_data, plot = False))
    return embeddings

# extract_embedding(random_audio)

#%%
# 'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
extracted_embeddings = []
for index_num,row in tqdm(metadata.iterrows()):
    try:
        file_name = os.path.join(os.path.abspath(DATASET_PATH),'train', str(row["primary_label"]) , str(row["filename"]))
        final_class_labels=row["primary_label"]
        data=extract_embedding(file_name)
        extracted_embeddings.append([data,final_class_labels])
    # Se o arquivo não for válido, pula e continua
    except ValueError:
        continue


# primary_label = []
# secondary_label = []
# type_ = []
# latitude = []
# longitude = []
# scientific_name = []
# common_name = []
# author = []
# date = []
# filename = []
# license_ = []
# rating = []
# time = []
# url = []
# duration = []
# split = []

# def create_dataset(DATASET_PATH):
#     for root, dirs, files in tqdm(os.walk(DATASET_PATH)):
#         for file in files:



#%%
'''Treinando o modelo'''
# Model Marker fornece pontuações de classe em nível de quadro (ou seja, 521 pontuações para cada quadro). Para determinar as previsões no nível do clipe, as pontuações podem ser agregadas por classe em todos os quadros (por exemplo, usando agregação média ou máxima). Isto é feito por abaixo scores_np.mean(axis=0) . Finalmente, para encontrar a classe com melhor pontuação no nível do clipe, você pega o máximo das 521 pontuações agregadas.
# Ao usar Model Marker para classificação de áudio, você deve começar com uma especificação do modelo. Este é o modelo básico do qual seu novo modelo extrairá informações para aprender sobre as novas classes. Também afeta como o conjunto de dados será transformado para respeitar os parâmetros de especificação do modelo, como: taxa de amostragem, número de canais.

# YAMNet é um classificador de eventos de áudio treinado no conjunto de dados AudioSet para prever eventos de áudio da ontologia AudioSet.

# Espera-se que sua entrada seja de 16kHz e com 1 canal.

# Você não precisa fazer nenhuma reamostragem. O modelo YAMNet cuida disso para você.

#  * frame_length é para decidir quanto tempo terá cada amostra de treinamento. O número de amostras em cada quadro de áudio. Se o arquivo de áudio for menor que frame_length, então o arquivo de áudio será ignorado. Neste caso podemos aplicar a seguinte fórmula: COMPRIMENTO_ESPERADO_DA_FORMA_DE_ONDA * 3 segundos
# * frame_steps é número de amostras entre dois quadros de áudio. Este valor deve ser maior que frame_length. Isto é usado para decidir a que distância estão as amostras de treinamento. Nesse caso, a iª amostra começará em COMPRIMENTO_ESPERADO_DA_FORMA_DE_ONDA * 6s após a (i-1)ª amostra.

# O motivo para definir esses valores é contornar algumas limitações no conjunto de dados do mundo real.

# Por exemplo, no conjunto de dados de pássaros, os pássaros não cantam o tempo todo. Eles cantam, descansam e cantam novamente, com ruídos intermediários. Ter um quadro longo ajudaria a capturar o canto, mas defini-lo muito longo reduzirá o número de amostras para treinamento.

#%%
spec = audio_classifier.YamNetSpec(
    keep_yamnet_and_custom_heads=True,
    frame_step=3 * audio_classifier.YamNetSpec.EXPECTED_WAVEFORM_LENGTH,
    frame_length=6 * audio_classifier.YamNetSpec.EXPECTED_WAVEFORM_LENGTH)

#%%
'''Carregando os dados'''
# O Model Maker tem a API para carregar os dados de uma pasta e tê-los no formato esperado para a especificação do modelo.

# A divisão de treinamento e teste são baseados nas pastas. O conjunto de dados de validação será criado como 20% da divisão do treinamento.

# Nota: O parâmetro cache = True é importante para tornar o treinamento mais rápido, mas também exigirá mais RAM para armazenar os dados. Para o conjunto de dados de pássaros, isso não é um problema, pois tem apenas 300 MB, mas se você usar seus próprios dados, deve prestar atenção a eles.

train_data = audio_classifier.DataLoader.from_folder(
    spec, os.path.join(DATASET_PATH, 'train'), cache=True)
train_data, validation_data = train_data.split(0.8)
test_data = audio_classifier.DataLoader.from_folder(
    spec, os.path.join(DATASET_PATH, 'test'), cache=True)


#%%
'''Criando o modelo'''
# model = Sequential()

# model.add(Input(shape=(1024), dtype=tf.float32, name='input_embedding'))
# model.add(Dense(512))
# model.add(Activation('relu'))

# model.add(Dense(len(bird_code_to_name)))

# model.summary()

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1024), dtype=tf.float32,
                          name='input_embedding'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(len(bird_code_to_name))
], name='model')

model.summary()

#%%
# Configure o modelo Keras com o otimizador Adam e a perda de entropia cruzada
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

#%%
# Treine o modelo em 10 épocas para fins de demonstração
EPOCHS = 10

# API Callbacks
# Um retorno de Callbacks é um objeto que pode realizar ações em vários estágios de treinamento (por exemplo, no início ou no final de uma época, antes ou depois de um único lote, etc.).
# Podemos usar callbacks para:
#  * Grave registros do TensorBoard após cada lote de treinamento para monitorar suas métricas
#  * Salve periodicamente seu modelo no disco
#  ... e mais
my_callbacks = (
  tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
  tf.keras.callbacks.ModelCheckpoint(filepath='saved_models/transfer_learn_YAMNET_BIRDS.hdf5', save_best_only=True)
)

history = model.fit(
    train_data, 
    validation_data=validation_data,  
    epochs=EPOCHS,
    callbacks=[my_callbacks],
)

# %%
