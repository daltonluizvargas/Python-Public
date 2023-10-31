#%%
'''
Existem muitos casos de uso importantes de classificação de áudio, incluindo para proteger a vida selvagem , para detectar baleias e até mesmo para lutar contra o desmatamento ilegal .

How ZSL uses ML to classify gunshots to protect wildlife
https://cloud.google.com/blog/products/ai-machine-learning/how-zsl-uses-google-cloud-to-analyse-acoustic-data

Acoustic Detection of Humpback Whales Using a Convolutional Neural Network
https://ai.googleblog.com/2018/10/acoustic-detection-of-humpback-whales.html

The fight against illegal deforestation with TensorFlow
https://blog.google/technology/ai/fight-against-illegal-deforestation-tensorflow/

'''
# pip install --user -r /path/to/requirements.txt
'''Importando as bibliotecas'''
import os
from datetime import datetime

import librosa
import librosa.display as ld
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from IPython.display import Audio

# tqdm é uma biblioteca em Python usada para criar medidores de progresso ou barras de progresso. O nome tqdm vem do nome árabe taqaddum, que significa 'progresso'
from tqdm import tqdm

import tensorflow as tf
print(tf.__version__)

from sklearn.metrics import classification_report

# Pré-processamento dos dados
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import (Activation, Conv1D, Dense, Dropout,
                                     Flatten, MaxPooling1D)
# Definição do modelo, camadas, função de ativação,...
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical

# Avaliação
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import seaborn as sns
sns.set() # sobrescrevemos a aparência 'mat plot lib' por 'sns.set ()', para tirar vantagem do estilo nativo do seaborn.
# Em essência, isso nos permitirá codificar os gráficos em Matplotlib, mas eles serão exibidos com o que alguns chamam de “aparência Seaborn muito superior”

#%%
'''Carregando o dataset'''
DATASET_PATH = 'Datasets/UrbanSound8K/audio/'
DATA = 'Datasets/UrbanSound8K/metadata/UrbanSound8K.csv'

# %%
metadata = pd.read_csv(DATA)
metadata.head()

#%%
fsID = [] # Freesound ID da gravação da qual este trecho (fatia) é obtido
classID = [] # identificador numérico da classe de som
occurID = [] # identificador numérico para distinguir diferentes ocorrências do som na gravação original
sliceID = [] # identificador numérico para distinguir diferentes fatias retiradas da mesma ocorrência

full_path = []

for root, dirs, files in tqdm(os.walk(DATASET_PATH)):
    for file in files:
        try:          
            fs = int(file.split('-')[0])
            class_ = int(file.split('-')[1])
            occur = int(file.split('-')[2])

            # Neste ID precisamos dividir 2x a string entre o separador com o traço e depois com o formato do arquivo .wav
            slice_ = file.split('-')[3]
            slice_ = int(slice_.split('.')[0])

            fsID.append(fs)
            classID.append(class_)
            occurID.append(occur)
            sliceID.append(slice_)

            full_path.append((root, file))
            
          # Se o arquivo não for válido, pula e continua
        except ValueError:
            continue
#%%
# Criamos um dicionário de sons (10)
# 0 = ar_condicionado, 1 = buzina_de_carro, 2 = crianca_brincando, 3 = latido_de_cachorro, 4 = perfuracao, 5 = motor_em_marcha_lenta, 6 = tiro_de_arma, 7 = britadeira, 8 = sirene, 9 = musica_de_rua
song_list = ['ar_condicionado', 'buzina_de_carro', 'crianca_brincando', 'latido_de_cachorro', 'perfuracao', 'motor_em_marcha_lenta', 'tiro_de_arma', 'britadeira', 'sirene', 'musica_de_rua']
song_dict = {em[0]:em[1] for em in enumerate(song_list)}

song_dict

#%%
# Visualizando os dados em um dataframe/tabela transposta (colunas viram linhas ou vice-versa --> função .T)
df = pd.DataFrame([fsID, classID, occurID, sliceID, full_path]).T

# %%
df
# %%
# Criando as colunas, dando nomes a elas
df.columns = ['fsID', 'classID', 'occurID', 'sliceID', 'path']

# Mapeando/Buscando a classe
df['classID'] = df['classID'].map(song_dict)
df['path'] = df['path'].apply(lambda x: x[0] + '/' + x[1])

df.head()

#%%
df.classID.value_counts()

# %%
# Contagem de registros por classe, iniciando classe com a maior quantidade de registro até a classe com a menor quantidade de registros
df.classID.value_counts().plot(kind='barh')

# %%
n_files = df.shape[0]

# Escolher um audio aleatório entre 0 e 1440 (número de registros)
rnd = np.random.randint(0, n_files)

rnd
#%%
# Usar o Librosa para carregar o arquivo aleatório de audio de acordo com o 
fname = df.path[rnd] 
fname
#%%
data, sampling_rate = librosa.load(fname, sr=44100)

plt.figure(figsize=(15, 5))
info = df.iloc[rnd].values
title_txt = f'Som: {info[1]}'
plt.title(title_txt.upper(), size=16)
librosa.display.waveplot(data, sr=sampling_rate)

# %%
X = librosa.stft(data)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(15, 5))
librosa.display.specshow(Xdb, sr=sampling_rate, x_axis='time', y_axis='hz')
plt.colorbar()
plt.title(title_txt.upper(), size=16)
plt.show()

# %%
'''Extraindo recursos/características MFCC's de cada arquivo de áudio do dataset'''
# Função extratora de recursos
def features_extractor(file_name):
    # Usamos o librosa para extrair:
    # Sample rate é o número de vezes por segundo em que as frequências são registradas
    # audio são os dados do áudio
    # res_type é o tipo de reamostragem de dados. Por padrão é usado o modo de alta qualidade Kaiser_best, mas neste exemplo iremos usar o Kaiser_fast por este é mais rápido (como o próprio nome diz)
    audio, sample_rate = librosa.load(file_name, res_type = 'kaiser_fast')
    
    # Extraimos os recursos da frequencia de MEL - MFCC, usando a função pronta do librosa
    mfccs_features = librosa.feature.mfcc(y = audio, sr = sample_rate, n_mfcc = 40)

    # A fim de obter o ponto médio, fazemos a transposição dos recursos extraídos
    mfccs_scaled_features = np.mean(mfccs_features.T, axis = 0)

    return mfccs_scaled_features

extracted_features=[]

print('[INFO] inciando a extração de recursos...')
for path in tqdm(df.path.values):
  data = features_extractor(path)
  extracted_features.append([data])

print('[INFO] extração de recursos concluída')


# %%
extracted_features

# %%
# Convertendo os recursos extraídos para visualização com Pandas
extracted_features_df = pd.DataFrame(extracted_features, columns = ['feature'])
# extracted_features_df.shape
extracted_features_df.head()


# %%
'''Dividindo entre atributos classe(class) e atributos previsores(features)'''
X = np.array(extracted_features_df['feature'].tolist())
y = np.array(df.classID.tolist())

# %%
X.shape

#%%
y.shape

# %%
y

#%%
X

#%%
'''Criando o modelo'''
# Transformando os valores categórios em números utilizando o LabelEncoder
# Assim para cada registro já calssificado, os codifica para um número entre 0 e 1
labelencoder = LabelEncoder()
y = to_categorical(labelencoder.fit_transform(y))
# %%
y

# %%
# Separando a base de dados em treinamento e teste
# Usamos 20% (0.2) para testar e 80% para treinar
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# %%
X_train

#%%
X_train.shape

# %%
X_test.shape

# %%
Y_train.shape

# %%
Y_test.shape

#%%
# Reshape para adicionar ao Tensor3D mais 1 dimensão
X_train = X_train[:,:,np.newaxis]
X_test = X_test[:,:,np.newaxis]
X_test.shape

# %%
# Número de classes
num_labels = y.shape[1]

# %%
num_labels

# %%
'''Criar a rede'''
# Número de camadas ocultas
# Neurônios = (Entradas + Saídas) / 2

# Entradas são os atributos previsores da base de dados
# Pesos: No treinamento de redes neurais o objeto é encontrar os pesos para cada um dos atributos
# https://iaexpert.academy/2020/05/25/funcoes-de-ativacao-definicao-caracteristicas-e-quando-usar-cada-uma/

# https://stackoverflow.com/questions/57387485/how-to-choose-units-for-dense-in-tensorflow-keras
# https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
model=Sequential()

model.add(Conv1D(64, kernel_size=(10), activation='relu',input_shape=(X_train.shape[1],1)))
model.add(Dropout(0.4))
model.add(MaxPooling1D(pool_size=(4)))

model.add(Conv1D(128, 10,padding='same',))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(MaxPooling1D(pool_size=(4)))

model.add(Flatten())

model.add(Dense(64))
model.add(Dropout(0.4))

model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')
model.summary()

# %%
'''Treinando o modelo'''
num_epochs = 80
num_batch_size = 32

checkpointer = ModelCheckpoint(filepath='saved_models/audio_classification.hdf5', 
                               verbose=1, save_best_only=True)
start = datetime.now()

model_history = model.fit(X_train, Y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, Y_test), callbacks=[checkpointer], verbose=1)


duration = datetime.now() - start
print("[INFO] treinamento concluído em: ", duration)

# %%
'''Avaliando o modelo'''
test_accuracy=model.evaluate(X_test,Y_test,verbose=0)

# test_accuracy[0] = valor da loss
# test_accuracy[1] = valor da accuracy
print(f"[INFO] accuracy: {test_accuracy[1]*100:.2f}%")

#%%
# As métricas são armazenadas em um dicionário no membro history do objeto retornado.
# listar as métricas coletadas
print(model_history.history.keys())

#%%
#%%
'''Exbir o hístórico de precisão e perda do modelo'''
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('results/AudioClassification_Model_Accuracy.png')
plt.show()

plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('results/AudioClassification_Model__Loss.png')
plt.show()

#%%
# Prevendo as classes
predictions = model.predict(X_test, batch_size=32)
predictions=predictions.argmax(axis=1)
predictions
predictions = predictions.astype(int).flatten()
predictions = (labelencoder.inverse_transform((predictions)))
predictions = pd.DataFrame({'Classes Previstas': predictions})

# Classes atuais da base de dados de teste, para combinar e comparar com o resultado das classes previstas pelo modelo
actual=Y_test.argmax(axis=1)
actual = actual.astype(int).flatten()
actual = (labelencoder.inverse_transform((actual)))
actual = pd.DataFrame({'Classes Reais': actual})

# Combinando
finaldf = actual.join(predictions)
finaldf[140:150]

#%%
cm = confusion_matrix(actual, predictions)
plt.figure(figsize = (12, 10))
cm = pd.DataFrame(cm , index = [i for i in labelencoder.classes_] , columns = [i for i in labelencoder.classes_])
ax = sns.heatmap(cm, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='')
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.title('Matriz de Confusão', size=20)
plt.xlabel('Classes Previstas', size=14)
plt.ylabel('Classes Reais', size=14)
plt.savefig('results/AudioClassification_Matriz_Confusão.png')
plt.show()

#%%
print(classification_report(actual, predictions))

# %%
'''Testando em um audio'''
AUDIO="testes/car_horn.wav"
#%%
audio, sample_rate = librosa.load(AUDIO, res_type='kaiser_fast') 
mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
mfccs_scaled_features = mfccs_scaled_features[:,:,np.newaxis]

predictions = model.predict(mfccs_scaled_features, batch_size=32)
predictions=predictions.argmax(axis=1)
predictions
predictions = predictions.astype(int).flatten()
predictions = (labelencoder.inverse_transform((predictions)))

predictions

#%%
# Plotar o formato do sinal de áudio (como ele se comporta ao longo do tempo) + a classe
plt.figure(figsize=(14,5))
plt.title('Tipo de som: ' + str(predictions[0]).upper(), size=16)
ld.waveplot(audio, sr=sample_rate)
