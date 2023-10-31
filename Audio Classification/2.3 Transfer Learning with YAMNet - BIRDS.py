#%%
import seaborn as sns

sns.set()

import csv
import glob
import os
# pathlib — Caminhos do sistema de arquivos orientado a objetos
# https://docs.python.org/pt-br/3.8/library/pathlib.html
import pathlib
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
DATASET_PATH = 'Datasets/small_birds_dataset'
DATA = 'Datasets/small_birds_dataset/metadata.csv'

data_dir = pathlib.Path(DATASET_PATH)

#%%
metadata = pd.read_csv(DATA)
metadata.head()

#%%
birds = np.array(tf.io.gfile.listdir(str(DATASET_PATH + '/train')))
print(f'[INFO] birds: ', birds)

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
for index in range(len(birds)):
    print(f'Nome do pássaro: {bird_code_to_name[birds[index]]}')
    display(Image(birds_images[birds[index]]))

#%%
# Nesta base de dados cada passaro/classes é definida por pasta/diretório

# tf.gfile é uma abstração para acessar o sistema de arquivos, ou seja,você pode ter uma única API como uma abstração para vários sistemas de armazenamento persistente diferente
def get_files(dir, info = False):
  filenames = tf.io.gfile.glob(str(DATASET_PATH) + '/'+ dir + '/*/*.wav')
  filenames = tf.random.shuffle(filenames)
  num_samples = len(filenames)

  if info:
    print(f'[INFO] total de registros: ', num_samples)
    print(f'[INFO] arquivo de exemplo: ', filenames[0])

  return filenames

get_files('train')

# %%
# Dividir os arquivos em conjuntos de treinamento, validação usando uma proporção de 80:20
train_files = get_files('train')[:193] # 80% para treinar
val_files = get_files('train')[193: 193 + 48] # 20% para validar
test_files = get_files('test')

print('[INFO] tamanho do conjunto de treinamento', len(train_files))
print('[INFO] tamanho do conjunto de validação', len(val_files))
print('[INFO] tamanho do conjunto de teste', len(test_files))

#%%
def load_wav_16k_mono(filename):
    """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav

testing_wav_data = load_wav_16k_mono(test_files[10])
_ = plt.plot(testing_wav_data)

# %%
# Os resultados do modelo são:
  # scores: previsões - pontuações para cada uma das 521 classes;
  # embeddings - recursos YAMNet;
  # spectrogram - espectrogramas
scores, embeddings, spectrogram = yamnet_model(testing_wav_data)
class_scores = tf.reduce_mean(scores, axis=0)
top_class = tf.argmax(class_scores)
infered_class = class_names[top_class]

print(f'[INFO] o som principal é: {infered_class}')
print(f'[INF] a forma dos embeddings: {embeddings.shape}')
# %%
# O atributo classe de cada arquivo WAV é seu diretório pai
def get_label(file_path):
  # Ao acessar o caminho/diretório de cada arquivo, precisamos pegar somente a classe a que ele corresponde, então usamos a função tf.strings.split passando o caminho completo até o arquivo e definindo um separador de caminhos com a função os.path.sep, ou seja, cada diretório estará separado dentro desta lista
  # Exemplo: 
  # Este é o path: Datasets\\mini_speech_commands\\yes\\8134f43f_nohash_4.wav
  # Usando um separador por diretórios do sistema (os.path.sep) temos a saída: 
  # [b'Datasets' b'mini_speech_commands' b'yes' b'8134f43f_nohash_4.wav']
  # [[-4]Datasets, [-3]mini_speech_commands, [-2]yes, [-1]8134f43f_nohash_4.wav]
  # Fazendo uma recursão nos indices de cada diretório, conseguimos retornar somente a classe de cada arquivo nos dois diretórios anteriores [-2] (diretório pai)
  parts = tf.strings.split(file_path, os.path.sep)

  return parts[-2]

# Teste
labels = []
for index in range(len(birds)):
  labels.append(get_label(train_files[index]))
  print(labels)
#%%
main_ds = tf.data.Dataset.from_tensor_slices(train_files, labels)
main_ds.element_spec
# %%
def load_wav_for_map(train_files):
  return load_wav_16k_mono(train_files)

main_ds = main_ds.map(load_wav_for_map)
main_ds.element_spec
# %%
