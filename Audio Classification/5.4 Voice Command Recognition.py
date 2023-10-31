#%%
# Built-in Python Modules 
# These are mainly for file operations
import glob
import os
import pathlib
import random

import pyaudio, wave
from scipy.io import wavfile

import librosa
import librosa.display as ld

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

import seaborn as sns
sns.set() # sobrescrevemos a aparência 'mat plot lib' por 'sns.set ()', para tirar vantagem do estilo nativo do seaborn.
# Em essência, isso nos permitirá codificar os gráficos em Matplotlib, mas eles serão exibidos com o que alguns chamam de “aparência Seaborn muito superior”.

# tqdm é uma biblioteca em Python usada para criar medidores de progresso ou barras de progresso. O nome tqdm vem do nome árabe taqaddum, que significa 'progresso'
from tqdm import tqdm

# O filtro de avisos controla se os avisos são ignorados, exibidos ou transformados em erros (gerando uma exceção). Neste exemplo vamos ignorá-los
import warnings
warnings.filterwarnings("ignore")

#%%
# Dataset directories
DATASET_MINI_SPEECH = pathlib.Path('Datasets/mini_speech_commands')
audio_dir = pathlib.Path(DATASET_MINI_SPEECH)

#%%
'''Entendendo a base de dados'''
# Diferente das bases de dados no exemplos anteriores, onde a classe é definida pela nomeclatura do arquivo, nesta base de dados cada classe é definida por pasta

# Visualizar os comandos básicas sobre o conjunto de dados
speech_commands = []

# Percorrer a lista de pastas para obter os comandos
for name in glob.glob(str(DATASET_MINI_SPEECH) + '/*' + os.path.sep):
  speech_commands.append(name.split('\\')[-2])

# Compresão de dicionário para mapear comandos para um inteiro para pesquisas rápidas
speech_commands_dict = {i : speech_commands.index(i) for i in speech_commands}
print('[INFO] comandos:', speech_commands_dict)

#%%
# Extrair os arquivos de áudio em uma lista e depois embaralha-los/misturá-los
# tf.io.gfile.glob retorna uma lista de strings contendo nomes de arquivos que correspondem ao padrão fornecido.
filenames = tf.io.gfile.glob(str(DATASET_MINI_SPEECH) + '/*/*')
filenames = tf.random.shuffle(filenames)
num_samples = len(filenames)
num_recordings = len(tf.io.gfile.listdir(str(DATASET_MINI_SPEECH/speech_commands[0])))
print('[INFO] total de arquivos:', num_samples)
print('[INFO] arquivos por comando:', num_recordings)
print('[INFO] arquivo de exemplo:', filenames[0])

#%%
# Carregar um arquivo de exemplo
sample_file = audio_dir/'yes/004ae714_nohash_0.wav'
samples, sample_rate = librosa.load(sample_file, sr = 16000)
fig = plt.figure(figsize=(14, 8))
plt.title('Onda sonora para o arquivo: ' + str(sample_file), size=16)
ld.waveplot(samples, sr=sample_rate)

#%%
# Número de gravações para cada comando de voz
plt.figure(figsize=(30,10))
index = np.arange(len(speech_commands))
plt.bar(index, num_recordings)
plt.xlabel('Comandos', fontsize=12)
plt.ylabel('Total de arquivos', fontsize=12)
plt.xticks(index, speech_commands, fontsize=15, rotation=60)
plt.title('No. de arquivos por comando')
plt.show()

#%%
# Visualizando a distribuição da duração das gravações, assim conseguimos visualizar que a duração de algumas gravações é inferior a 1 segundo
duration_of_recordings=[]
for label in speech_commands:
    waves = [f for f in os.listdir(DATASET_MINI_SPEECH/label) if f.endswith('.wav')]
    for wav in waves:
        sample_rate, samples = wavfile.read(DATASET_MINI_SPEECH/label/wav)
        duration_of_recordings.append(float(len(samples)/sample_rate))
    
plt.hist(np.array(duration_of_recordings))

#%%
speech_data_list = []

# Criar uma lista com todos os arqquivos de audio
for name in tqdm(glob.glob(str(DATASET_MINI_SPEECH) + '/*/*')):
  speech_data_list.append(name)

# Embaralhar a lista de arquivos de áudio
random.seed(42)
random.shuffle(speech_data_list)
#%%
# Criar uma lista de atributos classe de cada arquivo de áudio, armazenando o nome da pasta em que está armazenado. Lembrando que o nome da pasta é o atributo classe, por isso pegamos sempre o diretório anterior
speech_data_labels = []

for audio in tqdm(speech_data_list):
  speech_data_labels.append(os.path.dirname(audio).split('\\')[-1])

#%%
# Converter os atributos classe em inteiros (semelhante ao que o LabelEncoder faz)
speech_label_int = []

for audio in tqdm(speech_data_labels, colour='green'):
  speech_label_int.append(speech_commands_dict[audio])

# Exemplo
# speech_label_int[2990:3002]

#%%
# Compilando todos os dados de fala em uma lista e aplicando a normalização para 16000hz
loaded_speech_data = []

for audio in tqdm(speech_data_list, colour='green'):
  loaded_speech_data.append(librosa.load(audio, sr=16000))

#%%
# Extraindo os recursos MFCCs
speech_data_mfcc = []

for loaded_audio in tqdm(loaded_speech_data, colour='green'):
  speech_data_mfcc.append(librosa.feature.mfcc(loaded_audio[0], loaded_audio[1]))

#%%
example_index = 5
ld.specshow(speech_data_mfcc[example_index], x_axis='time', y_axis='hz')
plt.colorbar()
plt.tight_layout()
plt.title(f'MFCC para o arquivo \"{speech_data_labels[example_index]}\"')
plt.show

#%%
# Podemos fazer a inversão recursos MFCC, convertendo coeficientes cepstral de frequência Mel em um sinal de áudio no domínio do tempo
waveform_example = librosa.feature.inverse.mfcc_to_audio(
    speech_data_mfcc[example_index])
ld.waveplot(waveform_example)
plt.tight_layout()
plt.title(f'Forma da onda para \"{speech_data_labels[example_index]}\"')
plt.show

#%%
# Separar os dados em:
# 70% para treinar
# 15% para validar
# 15% para testar
data_length = len(speech_data_list)
data_ratio = {
    'train': 0.7,
    'validate': 0.15,
    'test': 0.15
}
training_ratio = int(data_length*data_ratio['train'])
validation_ratio = int(data_length*data_ratio['validate'])
testing_ratio = int(data_length*data_ratio['test'])

print(f"Proporção do conjunto de dados")
print(f"[INFO] Dados para treinamento: {training_ratio:.0f}")
print(f"[INFO] Dados para validação: {validation_ratio:.0f}")
print(f"[INFO] Dados para teste: {testing_ratio:.0f}")


#%%
speech_data_as_tensor = []

for index in range(len(speech_data_mfcc)):
  # Precisamos fazer um redimensionamento para corrigir a inconsistência no tamanho da matriz e ser aceita nas dimensões da rede neural convolucional 2D
  mfcc_array = np.copy(speech_data_mfcc[index])
  mfcc_array.resize((20,32), refcheck=False)
  speech_data_as_tensor.append(tf.expand_dims(
      tf.convert_to_tensor(mfcc_array), -1))

speech_data_as_tensor[0].shape

#%%
# Fatiar o conjunto de dados para as proporções desejadas que visualizamos nos passos anteriores
training_slice = speech_data_as_tensor[:5600]
validation_slice = speech_data_as_tensor[5600: 5600 + 1200]
testing_slice = speech_data_as_tensor[5600 + 1200:]

#%%
# tf.data.Dataset.from_tensor_slices() obtemos as fatias de um array na forma de objetos, ou seja, cada elemento da lista será retornado separadamente
training_dataset = tf.data.Dataset.from_tensor_slices((
    training_slice, speech_label_int[:5600]))
# for training in training_dataset: # Demonstração do que acontece (vai demorar mais para processar)
#   print(training)
validation_dataset = tf.data.Dataset.from_tensor_slices((
    validation_slice, speech_label_int[5600: 5600+1200]))
testing_dataset = tf.data.Dataset.from_tensor_slices((
    testing_slice, speech_label_int[-1200:]))

batch_size = 10

# dataset.batch combina elementos consecutivos deste conjunto de dados em lotes, em outras palavras, pegará as primeiras 10 entradas e fará um lote com elas e assim por diante com os demais registros criando lotes com o tamanho 10.
# https://www.gcptutorials.com/article/how-to-use-batch-method-in-tensorflow
training_dataset = training_dataset.batch(batch_size)
validation_dataset = validation_dataset.batch(batch_size)

#%%
num_labels = len(speech_commands)

# NORMALIZAÇÃO
# norm_layer.adapt irá calcular a média e a variância dos dados e armazená-los como pesos da camada.
norm_layer = layers.Normalization()

#https://learnopencv.com/image-classification-using-convolutional-neural-networks-in-keras/
# Cria a rede neural convolucional
model = models.Sequential([
    layers.Input(shape=(20,32,1)),
    # layers.Resizing: em nosso modelo sequencial, adicionamos uma camada de pré-processamento que redimensiona as imagens para um tamanho menor (32x32) para permitir que o modelo treine mais rápido.
    layers.Resizing(32, 32),

    # Normalização 
    norm_layer,
    # Camada de convolução 2D com filtro inicial de 32 e depois 64, ambas com um kernel 3x3
    # O primeiro parâmetro Conv2D necessário é o número de filtros que a camada convolucional aprenderá.
    # As camadas Conv2D intermediárias aprenderão mais filtros do que as primeiras camadas Conv2D, mas menos filtros do que as camadas mais próximas da saída.
    # O segundo parâmetro obrigatório que você precisa fornecer para a classe Keras Conv2D é o kernel_size , uma tupla especificando a largura e a altura da janela de convolução 2D ou pode ser um único inteiro para especificar o mesmo valor para todas as dimensões espaciais.
    # https://www.pyimagesearch.com/2018/12/31/keras-conv2d-and-convolutional-layers/
    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(64, 3, activation='relu'),

    #O MaxPooling2D é então usado para reduzir o tamanho espacial (apenas largura e altura, não profundidade). Isso reduz o número de parâmetros, portanto, o cálculo é reduzido. Usar menos parâmetros evita overfitting.
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_labels),
])
model.summary()

#%%
# Configure o modelo Keras com o otimizador Adam e a perda de entropia cruzada
model.compile(
    optimizer='Adam',
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
  tf.keras.callbacks.ModelCheckpoint(filepath='saved_models/voice_command_recognition.hdf5', save_best_only=True)
)
history = model.fit(
    training_dataset, 
    validation_data=validation_dataset,  
    epochs=EPOCHS,
    callbacks=[my_callbacks],
)

#%%
# Vamos plotar as curvas de perda de treinamento e validação para verificar como o modelo melhorou durante o treinamento
metrics = history.history
plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# plt.savefig('results/VoiceCommandRecognition_Model_Loss.png')
plt.show()

plt.plot(history.epoch, metrics['accuracy'], metrics['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# plt.savefig('results/VoiceCommandRecognition_Model_Accuracy.png')
plt.show()

#%%
test_audio_data = []
test_label_data = []

for audio, label in testing_dataset:
  test_audio_data.append(audio.numpy())
  test_label_data.append(label.numpy())

test_audio_data = np.array(test_audio_data)
test_label_data = np.array(test_label_data)

predicted_values = np.argmax(model.predict(test_audio_data), axis=1)
true_values = test_label_data

test_accuracy = sum(predicted_values == true_values) / len(true_values)
print(f'[INFO] precisão no conjunto de teste: {test_accuracy:.0%}')

#%%
'''Exibir uma matriz de confusão'''
# Use uma matriz de confusão para verificar quão bem o modelo classifica cada um dos comandos do conjunto de teste
confusion_mtx = tf.math.confusion_matrix(true_values, predicted_values) 
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx, xticklabels=speech_commands_dict, yticklabels=speech_commands_dict, 
            annot=True, fmt='g', cmap='coolwarm')
plt.xlabel('Classes Previstas')
plt.ylabel('Calsses Reais')
plt.show()


#%%
'''Executar inferência em um arquivo de áudio'''
# Finalmente, verifique a saída de previsão do modelo usando um arquivo de áudio de entrada de alguém dizendo "não"
def predict(sample_file, plot = False):
  audio, sample_rate = librosa.load(sample_file, sr = 16000)
  mfcc = librosa.feature.mfcc(audio)
  
  # A inconsistência no tamanho da matriz é corrigida redimensionando amatriz e   preenchendo com zeros 
  mfcc_array = np.copy(mfcc)
  mfcc_array.resize((20,32), refcheck=False)
  speech_tensor = tf.expand_dims(tf.convert_to_tensor(mfcc_array), -1)
  speech_tensor = tf.expand_dims(speech_tensor, axis=0)
  
  prediction = model(speech_tensor)
  index = np.argmax(prediction[0])
  result = f' Comando previsto "{speech_commands[index]}"'
  if plot:
    plt.bar(speech_commands, tf.nn.softmax(prediction[0]), color=list('rgbkymc')  )
    plt.title(str(result).upper())
    plt.show()
  
  else:
    print('[INFO]' + result)

WAVE_OUTPUT_FILENAME = 'recordings/Audio_.wav'
def recordAudio(record_seconds = 10):
    CHUNK = 8192
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = record_seconds
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
             channels = CHANNELS,
             rate = RATE,
             input = True,
             input_device_index = 0,
             frames_per_buffer = CHUNK)
    print("[INFO]* gravando")
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("[INFO]* gravação concluída")
    stream.stop_stream()    # "Stop Audio Recording
    stream.close()          # "Close Audio Recording
    p.terminate()           # "Audio System Close
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

print('[INFO] funções criadas')

#%%
filenames = tf.io.gfile.glob(str(DATASET_MINI_SPEECH) + '/*/*')
n_files = len(filenames)

rnd = np.random.randint(0, n_files)
filenames[rnd]
fname = filenames[rnd] 
command = filenames[rnd]
print(f'[INFO] arquivo de audio No. {rnd}')
print(f'[INFO] path: {fname}')
predict(fname, plot = True)

#%%
# '''Carregar o modelo criado e testar em audio gravado'''
# model = tf.keras.models.load_model('saved_models/voice_command_recognition_testando_novo.hdf5')

# recordAudio(record_seconds=2)
# predict(WAVE_OUTPUT_FILENAME)


