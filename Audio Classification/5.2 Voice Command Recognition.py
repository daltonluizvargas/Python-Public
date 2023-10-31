#%%
import os
from datetime import datetime

import librosa
import librosa.display as ld
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# pathlib — Caminhos do sistema de arquivos orientado a objetos
# https://docs.python.org/pt-br/3.8/library/pathlib.html
import pathlib

from scipy.io import wavfile

import tensorflow as tf

# Pré-processamento dos dados
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Modelo, camadas, ...
from tensorflow.keras.layers import (Activation, Conv1D, Dense, Dropout, Flatten, MaxPooling1D)
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint

# Avaliação
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# tqdm é uma biblioteca em Python usada para criar medidores de progresso ou barras de progresso. O nome tqdm vem do nome árabe taqaddum, que significa 'progresso'
from tqdm import tqdm

import seaborn as sns
sns.set() # sobrescrevemos a aparência 'mat plot lib' por 'sns.set ()', para tirar vantagem do estilo nativo do seaborn.
# Em essência, isso nos permitirá codificar os gráficos em Matplotlib, mas eles serão exibidos com o que alguns chamam de “aparência Seaborn muito superior”.

import warnings
warnings.filterwarnings("ignore")

#%%
DATASET_MINI_SPEECH = pathlib.Path('Datasets/mini_speech_commands')
data_dir = pathlib.Path(DATASET_MINI_SPEECH)

#%%
# Diferente das bases de dados no exemplos anteriores, onde a classe é definida pela nomeclatura do arquivo, nesta base de dados cada classe é definida por pasta

# Visualizar os comandos básicas sobre o conjunto de dados
# Cada comando/classes é separado por pastas
# tf.gfile é uma abstração para acessar o sistema de arquivos, ou seja,você pode ter uma única API como uma abstração para vários sistemas de armazenamento persistente diferentes
commands = np.array(tf.io.gfile.listdir(str(DATASET_MINI_SPEECH)))
commands = commands[commands != 'README.md']
print('[INFO] commands:', commands)

#%%
# Extrair os arquivos de áudio em uma lista e depois embaralha-los/misturá-los
# tf.io.gfile.glob retorna uma lista de strings contendo nomes de arquivos que correspondem ao padrão fornecido.
filenames = tf.io.gfile.glob(str(DATASET_MINI_SPEECH) + '/*/*')
filenames = tf.random.shuffle(filenames)
num_samples = len(filenames)
num_recordings = len(tf.io.gfile.listdir(str(DATASET_MINI_SPEECH/commands[0])))
print('[INFO] total de arquivos:', num_samples)
print('[INFO] arquivos por comando:', num_recordings)
print('[INFO] arquivo de exemplo:', filenames[0])

#%%
# Carregar um arquivo de exemplo
sample_file = data_dir/'yes/004ae714_nohash_0.wav'
samples, sample_rate = librosa.load(sample_file, sr = 16000)
fig = plt.figure(figsize=(14, 8))
plt.title('Onda sonora para o arquivo: ' + str(sample_file), size=16)
ld.waveplot(samples, sr=sample_rate)

#%%
# Visualizar a taxa de amostragem do sinal de áudio
ipd.Audio(samples, rate=sample_rate)
print(sample_rate)

#%%
# Número de gravações para cada comando de voz
plt.figure(figsize=(30,10))
index = np.arange(len(commands))
plt.bar(index, num_recordings)
plt.xlabel('Comandos', fontsize=12)
plt.ylabel('Total de arquivos', fontsize=12)
plt.xticks(index, commands, fontsize=15, rotation=60)
plt.title('No. de arquivos por comando')
plt.show()

#%%
# Visualizando a distribuição da duração das gravações, assim podemos visualizar que a duração de algumas gravações é inferior a 1 segundo
duration_of_recordings=[]
for label in commands:
    waves = [f for f in os.listdir(DATASET_MINI_SPEECH/label) if f.endswith('.wav')]
    for wav in tqdm(waves, desc = label):
        sample_rate, samples = wavfile.read(DATASET_MINI_SPEECH/label/wav)
        duration_of_recordings.append(float(len(samples)/sample_rate))
    
plt.hist(np.array(duration_of_recordings))

#%%
'''Carregando a base de dados'''
all_wave = []
all_label = []
path = []
for label in commands:
    waves = [f for f in os.listdir(DATASET_MINI_SPEECH/label) if f.endswith('.wav')]
    for wav in tqdm(waves, desc = label):
        samples, sample_rate = librosa.load(DATASET_MINI_SPEECH/label/wav, sr  = 16000)
        all_wave.append(samples)
        all_label.append(label)  
        path.append(DATASET_MINI_SPEECH/label/wav)  

print("[INFO] base de dados carregada...")

#%%
# 
# Visualizando os dados em um dataframe/tabela transposta (colunas viram linhas ou vice-versa --> função .T)
df = pd.DataFrame([all_label, all_wave, path]).T

#%%
# Criando as colunas, dando nome a elas
df.columns = ['command', 'waves', 'path']
df.head(10)

#%%
# Tamanho do dataframe
df.shape[0]

#%%
# Contagem de registros por classe
df.command.value_counts()

#%%
# Visualizando a distribuição das gravações por classe/comando
df.command.value_counts().plot(kind = 'barh')

#%%
n_files = df.shape[0]

# Escolher um audio aleatório entre 0 e 8000 (número de registros)
rnd = np.random.randint(0, n_files)

fname = df.path[rnd] 
command = df.command[rnd]
print(f'[INFO] arquivo de audio No. {rnd}')
print(f'[INFO] path: {fname}')
print(f'[INFO] label: {command}')

#%%
# Visualizar a forma da onda
data, sampling_rate = librosa.load(fname, sr=44100)

plt.figure(figsize=(15, 5))
info = df.iloc[rnd].values
title_txt = f'Comando: {info[0]} - Arquivo: {info[2]}'
plt.title(title_txt.upper(), size=16)
librosa.display.waveplot(data, sr=sampling_rate)

#%%
# Visualizar a Transformada de Fourier para este arquivo de áudio
X = librosa.stft(data)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(15, 5))
librosa.display.specshow(Xdb, sr=sampling_rate, x_axis='time', y_axis='hz')
plt.colorbar()
plt.title(title_txt.upper(), size=16)
plt.show()

#%%
'''Extraindo recursos/características MFCC's de cada arquivo de áudio do dataset'''
# Função extratora de recursos
def features_extractor(file_name):
    # Usamos o librosa para extrair:
    # Sample rate é o número de vezes por segundo em que as frequências são registradas
    # audio são os dados do áudio
    # res_type é o tipo de reamostragem de dados. Por padrão é usado o modo de alta qualidade Kaiser_best, mas neste exemplo iremos usar o Kaiser_fast por este é mais rápido (como o próprio nome diz)
    audio, sample_rate = librosa.load(file_name, sr = 16000, res_type = 'kaiser_fast')
    
    # Extraimos os recursos da frequencia de MEL - MFCC, usando a função pronta do librosa
    mfccs_features = librosa.feature.mfcc(y = audio, sr = sample_rate, n_mfcc = 40)

    # A fim de obter o ponto médio, fazemos a transposição dos recursos extraídos
    mfccs_scaled_features = np.mean(mfccs_features.T, axis = 0)

    return mfccs_scaled_features

extracted_features=[]

for path in tqdm(df.path.values):
  data = features_extractor(path)
  extracted_features.append([data])

#%%
extracted_features

# %%
# Convertendo os recursos extraídos para visualização com Pandas
extracted_features_df = pd.DataFrame(extracted_features, columns = ['feature'])
# extracted_features_df.shape
extracted_features_df.head()

# %%
'''Dividindo entre atributos classe(class) e atributos previsores(features)'''
X = np.array(extracted_features_df['feature'].tolist())
y = np.array(df.command.tolist())
# %%
X
# %%
y
#%%
'''Criando o modelo'''
# Transformando os valores categórios em números utilizando o LabelEncoder
# Assim para cada registro já calssificado, os codifica para um número entre 0 e 1
labelencoder = LabelEncoder()
y = to_categorical(labelencoder.fit_transform(y))
y

#%%
y[4]

#%%
classes = list(labelencoder.classes_)
classes

# %%
# Separando a base de dados em treinamento e teste
# Usamos 20% (0.2) para testar e 80% para treinar
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# %%
X_train

# %%
y

#%%
X_train.shape[1]

# %%
X_test.shape

# %%
Y_train.shape

# %%
Y_test.shape

# %%
# Número de classes
num_labels = y.shape[1]

# %%
num_labels

#%%
# RESHAPE PARA ADICIONAR AO TENSOR 3D PARA A ENTRADA 1D DA CNN
X_train = X_train[:,:,np.newaxis]
X_test = X_test[:,:,np.newaxis]
X_test.shape

#%%
'''Criar a rede'''
model=Sequential()

# Camada Conv 1
model.add(Conv1D(32, kernel_size=(10), strides=1, padding='valid', activation='relu',input_shape=(X_train.shape[1],1)))

# Camada Conv 2
model.add(Conv1D(64, kernel_size=(10), strides=1, padding='valid', activation='relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(Dropout(0.25))
model.add(MaxPooling1D(pool_size=(4)))

# Camada Conv 3
model.add(Conv1D(128, kernel_size=(10), strides=1, padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(MaxPooling1D(pool_size=(4)))

# Camada Flatten
model.add(Flatten())

# Camada Densa 1
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Camada Densa 2
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Camada Densa de saída
model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')
model.summary()

#%%
'''Treinando o modelo'''
num_epochs = 100
num_batch_size = 32

checkpointer = ModelCheckpoint(filepath='saved_models/voice_command_recognition_novo.hdf5', 
                               verbose=1, save_best_only=True)
                               
start = datetime.now()


model_history = model.fit(X_train, Y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, Y_test), callbacks=[checkpointer], verbose=1)

duration = datetime.now() - start
print("[INFO] treinamento concluído em: ", duration)

# %%
'''Avaliando o modelo'''
test_accuracy=model.evaluate(X_test,Y_test,verbose=0)
print(f"[INFO] accuracy: {test_accuracy[1]*100:.2f}%")

# #%%
# # Vamos plotar as curvas de perda de treinamento e validação para verificar como o modelo melhorou durante o treinamento
# metrics = history.history
# plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# # plt.savefig('results/VoiceCommandRecognition_Model_Loss.png')
# plt.show()

# plt.plot(history.epoch, metrics['accuracy'], metrics['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# # plt.savefig('results/VoiceCommandRecognition_Model_Accuracy.png')
# plt.show()

# #%%
# '''Avaliando o modelo'''
# test_accuracy=model.evaluate(X_test,Y_test,verbose=0)
# print(f"[INFO] accuracy: {test_accuracy[1]*100:.2f}%")

# #%%
# # Prevendo as classes
# predictions = model.predict(X_test, batch_size=32)
# predictions=predictions.argmax(axis=1)
# predictions
# predictions = predictions.astype(int).flatten()
# predictions = (le.inverse_transform((predictions)))
# predictions = pd.DataFrame({'Classes Previstas': predictions})

# # Classes atuais da base de dados de teste, para combinar e comparar com o resultado das classes previstas pelo modelo
# actual=Y_test.argmax(axis=1)
# actual = actual.astype(int).flatten()
# actual = (le.inverse_transform((actual)))
# actual = pd.DataFrame({'Classes Reais': actual})

# # Combinando
# finaldf = actual.join(predictions)
# finaldf[140:150]

# #%%
# '''Exibir uma matriz de confusão'''
# # Use uma matriz de confusão para verificar quão bem o modelo classifica cada um dos comandos do conjunto de teste
# cm = confusion_matrix(actual, predictions)
# plt.figure(figsize = (12, 10))
# cm = pd.DataFrame(cm , index = [i for i in le.classes_] , columns = [i for i in le.classes_])
# ax = sns.heatmap(cm, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='g')
# bottom, top = ax.get_ylim()
# ax.set_ylim(bottom + 0.5, top - 0.5)
# plt.title('Matriz de Confusão', size=20)
# plt.xlabel('Classes Previstas', size=14)
# plt.ylabel('Classes Reais', size=14)
# # plt.savefig('results/EmotionRecognition_Matriz_Confusão.png')
# plt.show()

# #%%
# print(classification_report(actual, predictions))

# #%%
# '''Carregar o modelo'''
# model = tf.keras.models.load_model('saved_models/voice_command_recognition_novo.hdf5')
# WAVE_OUTPUT_FILENAME = 'recordings/Audio_.wav'

# def predict(audio):
#     prediction=model.predict(audio)
#     index=np.argmax(prediction[0])
#     return classes[index]

# print('[INFO] funções criadas')

# #%%
# # import random

# # index=random.randint(0,len(X_test)-1)
# # samples=X_test[index].ravel()
# # print(index)
# # print("Audio:",classes[np.argmax(Y_test[index])])

# # ipd.Audio(samples, rate=8000)
# audio = 'Datasets/mini_speech_commands/left/1b4c9b89_nohash_3.wav'

# samples, sample_rate = librosa.load(audio, sr = 16000)
# samples = librosa.resample(samples, sample_rate, 8000)
# samples=samples.reshape(1,-1)
# samples = samples[:,:,np.newaxis]
# print(predict(samples))
