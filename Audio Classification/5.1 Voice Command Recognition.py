#%%
import os
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
import pyaudio, wave

import tensorflow as tf

# Pré-processamento dos dados
from sklearn.preprocessing import LabelEncoder

# Modelo, camadas, ...
from tensorflow.keras.layers import (Conv1D, Dense, Dropout, Flatten, MaxPooling1D, Input)
from keras.models import Model
from keras.utils import np_utils

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

# %%
# Reamostragem - Podemos entender que a taxa de amostragem do sinal é de 16.000 hz. Vamos reamostrar para 8000 hz, já que a maioria das frequências relacionadas à fala está presente em 8000z.
samples = librosa.resample(samples, sample_rate, 8000)
ipd.Audio(samples, rate=8000)

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
# Voisualizando a distribuição da duração das gravações, assim conseguimos visualizar que a duração de algumas gravações é inferior a 1 segundo
duration_of_recordings=[]
for label in commands:
    waves = [f for f in os.listdir(DATASET_MINI_SPEECH/label) if f.endswith('.wav')]
    for wav in waves:
        sample_rate, samples = wavfile.read(DATASET_MINI_SPEECH/label/wav)
        duration_of_recordings.append(float(len(samples)/sample_rate))
    
plt.hist(np.array(duration_of_recordings))

#%%
'''Pré-processamento'''
# Reamostragem para 8000hz
all_wave = []
all_label = []
removed_wave = []
for label in commands:
    waves = [f for f in os.listdir(DATASET_MINI_SPEECH/label) if f.endswith('.wav')]
    for wav in tqdm(waves, desc = label):
        samples, sample_rate = librosa.load(DATASET_MINI_SPEECH/label/wav, sr = 16000)
        samples = librosa.resample(samples, sample_rate, 8000)
        if(len(samples) == 8000): 
            all_wave.append(samples)
            all_label.append(label)        
        else:
            removed_wave.append(samples)

print('[INFO] reamostragem concluída')
print(f'[INFO] total de arquivos reamostrados: {len(all_wave)}')
print(f'[INFO] total de arquivos removidos (< 1 sec.): {len(removed_wave)}')

#%%
# Converter as classes/comandos de saída em inteiros codificados utilizando LabelEncoder
le = LabelEncoder()
y=le.fit_transform(all_label)
classes= list(le.classes_)

#%%
classes

#%%
y

#%%
len(y)

#%%
# Converter os rótulos codificados inteiros em um vetor one-hot, pois é um problema de multiclassificação
y=np_utils.to_categorical(y, num_classes=len(commands))
y

#%%
# Remodele a matriz 2D para 3D, pois a entrada para o conv1d deve ser uma matriz 3D
# all_wave = np.array(all_wave).reshape(-1,8000,1)
# all_wave.shape

#%%
# Dividido em conjunto de treinamento e validação - Em seguida, treinaremos o modelo em 80% dos dados e validaremos nos 20% restantes
X = np.array(all_wave)
y = np.array(y)
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
num_labels

#%%
# RESHAPE PARA ADICIONAR AO TENSOR 3D PARA A ENTRADA 1D DA CNN
X_train = X_train[:,:,np.newaxis]
X_test = X_test[:,:,np.newaxis]
X_test.shape

#%%
inputs = Input(shape=(8000,1))

#First Conv1D layer
conv = Conv1D(8,13, padding='valid', activation='relu', strides=1)(inputs)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

#Second Conv1D layer
conv = Conv1D(16, 11, padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

#Third Conv1D layer
conv = Conv1D(32, 9, padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

#Fourth Conv1D layer
conv = Conv1D(64, 7, padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

#Flatten layer
conv = Flatten()(conv)

#Dense Layer 1
conv = Dense(256, activation='relu')(conv)
conv = Dropout(0.3)(conv)

#Dense Layer 2
conv = Dense(128, activation='relu')(conv)
conv = Dropout(0.3)(conv)

outputs = Dense(num_labels, activation='softmax')(conv)

model = Model(inputs, outputs)
model.summary()

#%%
# Configure o modelo Keras com o otimizador Adam e a perda de entropia cruzada
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#%%
# Treine o modelo em 10 épocas para fins de demonstração
EPOCHS = 20
# API Callbacks
# Um retorno de Callbacks é um objeto que pode realizar ações em vários estágios de treinamento (por exemplo, no início ou no final de uma época, antes ou depois de um único lote, etc.).
# Podemos usar callbacks para:
#  * Grave registros do TensorBoard após cada lote de treinamento para monitorar suas métricas
#  * Salve periodicamente seu modelo no disco
#  ... e mais
my_callbacks = (
  tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
  tf.keras.callbacks.ModelCheckpoint(filepath='saved_models/voice_command_recognition_novo.hdf5', save_best_only=True)
)
history = model.fit(
    X_train, Y_train,  
    validation_data=(X_test, Y_test),  
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
'''Avaliando o modelo'''
test_accuracy=model.evaluate(X_test,Y_test,verbose=0)
print(f"[INFO] accuracy: {test_accuracy[1]*100:.2f}%")

#%%
# Prevendo as classes
predictions = model.predict(X_test, batch_size=32)
predictions=predictions.argmax(axis=1)
predictions
predictions = predictions.astype(int).flatten()
predictions = (le.inverse_transform((predictions)))
predictions = pd.DataFrame({'Classes Previstas': predictions})

# Classes atuais da base de dados de teste, para combinar e comparar com o resultado das classes previstas pelo modelo
actual=Y_test.argmax(axis=1)
actual = actual.astype(int).flatten()
actual = (le.inverse_transform((actual)))
actual = pd.DataFrame({'Classes Reais': actual})

# Combinando
finaldf = actual.join(predictions)
finaldf[140:150]

#%%
'''Exibir uma matriz de confusão'''
# Use uma matriz de confusão para verificar quão bem o modelo classifica cada um dos comandos do conjunto de teste
cm = confusion_matrix(actual, predictions)
plt.figure(figsize = (12, 10))
cm = pd.DataFrame(cm , index = [i for i in le.classes_] , columns = [i for i in le.classes_])
ax = sns.heatmap(cm, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='g')
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.title('Matriz de Confusão', size=20)
plt.xlabel('Classes Previstas', size=14)
plt.ylabel('Classes Reais', size=14)
# plt.savefig('results/EmotionRecognition_Matriz_Confusão.png')
plt.show()

#%%
print(classification_report(actual, predictions))

#%%
'''Carregar o modelo'''
model = tf.keras.models.load_model('saved_models/voice_command_recognition_novo.hdf5')
WAVE_OUTPUT_FILENAME = 'recordings/Audio_.wav'

def predict(audio):
    prediction=model.predict(audio)
    index=np.argmax(prediction[0])
    return classes[index]

print('[INFO] funções criadas')

#%%
# import random

# index=random.randint(0,len(X_test)-1)
# samples=X_test[index].ravel()
# print(index)
# print("Audio:",classes[np.argmax(Y_test[index])])

# ipd.Audio(samples, rate=8000)
audio = 'Datasets/mini_speech_commands/left/1b4c9b89_nohash_3.wav'

samples, sample_rate = librosa.load(audio, sr = 16000)
samples = librosa.resample(samples, sample_rate, 8000)
samples=samples.reshape(1,-1)
samples = samples[:,:,np.newaxis]
print(predict(samples))
