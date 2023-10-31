#%%
'''Importando as bibliotecas'''
import os
from datetime import datetime

import librosa
import librosa.display as ld
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from IPython.display import Audio
# tqdm é uma biblioteca em Python usada para criar medidores de progresso ou barras de progresso. O nome tqdm vem do nome árabe taqaddum, que significa 'progresso'
from tqdm import tqdm

print(tf.__version__)

import seaborn as sns
# Avaliação
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
# Pré-processamento dos dados
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import (Activation, Conv1D, Dense, Dropout,
                                     Flatten, MaxPooling1D)
# Definição do modelo, camadas, função de ativação,...
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model

from sklearn.metrics import classification_report

sns.set() # sobrescrevemos a aparência 'mat plot lib' por 'sns.set ()', para tirar vantagem do estilo nativo do seaborn.
# Em essência, isso nos permitirá codificar os gráficos em Matplotlib, mas eles serão exibidos com o que alguns chamam de “aparência Seaborn muito superior”.

#%%
'''Carregando os dataset'''

TESS = 'Datasets/Tess/'
RAVDESS_SPEECH = 'Datasets/Ravdess/Audio_Speech_Actors_01-24/'
RAVDESS_SONG = 'Datasets/Ravdess/Audio_Song_Actors_01-24/'

#%%
dir_list = os.listdir(TESS)
dir_list.sort()
dir_list

#%%
path = []
emotion = []

for i in dir_list:
    fname = os.listdir(TESS + i)
    for f in fname:
#         if i == 'OAF_angry' or i == 'YAF_angry':
#             emotion.append('nervosa')
#         elif i == 'OAF_disgust' or i == 'YAF_disgust':
#             emotion.append('nojo')
#         elif i == 'OAF_Fear' or i == 'YAF_fear':
#             emotion.append('medo')
#         elif i == 'OAF_happy' or i == 'YAF_happy':
#             emotion.append('feliz')
#         elif i == 'OAF_neutral' or i == 'YAF_neutral':
#             emotion.append('neutra')                                
#         elif i == 'OAF_Pleasant_surprise' or i == 'YAF_pleasant_surprised':
#             emotion.append('surpreso')               
#         elif i == 'OAF_Sad' or i == 'YAF_sad':
#             emotion.append('triste')
        
#         path.append(TESS + i + "/" + f)

# TESS_df = pd.DataFrame(emotion, columns = ['emotion_label'])
# TESS_df['source'] = 'TESS'
# TESS_df['actors'] = 'feminino'
# TESS_df = pd.concat([TESS_df, pd.DataFrame(path, columns = ['path'])],axis=1)
# TESS_df.emotion_label.value_counts()

#%%
TESS_df.head()

#%%
modality = [] # Modalidade (01 = AV completo, 02 = apenas vídeo, 03 = apenas áudio).
voc_channel = [] # Canal vocal (01 = fala, 02 = música).
emotion = [] # Emoção (01 = neutro, 02 = calma, 03 = feliz, 04 = triste, 05 = zangado, 06 = com medo, 07 = nojo, 08 = surpreso).
intensity = [] # Intensidade emocional (01 = normal, 02 = forte). NOTA: Não há intensidade forte para a emoção 'neutra'.
phrase =[] # Frase (01 = "Crianças conversam perto da porta", 02 = "Cachorros estão sentados na porta").
actors = [] # Ator (01 a 24. Os atores com números ímpares são homens, os atores com números pares são mulheres)

full_path = []

def create_dataset(DATASET):  
  for root, dirs, files in tqdm(os.walk(DATASET)):
      for file in files:
          try:          
              modal = int(file[1:2])
              vchan = int(file[4:5])
              lab = int(file[7:8])
              ints = int(file[10:11])
              phr = int(file[13:14])
              act = int(file[19:20])

              modality.append(modal)
              voc_channel.append(vchan)
              emotion.append(lab) #only labels
              intensity.append(ints)
              phrase.append(phr)
              actors.append(act)

              full_path.append((root, file))

            # Se o arquivo não for válido, pula e continua
          except ValueError:
              continue

print("[INFO] carregando o conjunto de dados SPEECH...")
create_dataset(RAVDESS_SPEECH)
print("[INFO] carregando o conjunto de dados SONG...")
create_dataset(RAVDESS_SONG)

# Criamos um dicionário de emoções (8), ajutando-a para iniciar a partir do índice 1 ao 8, ficando desta maneira:
# 1 = neutra, 2 = calma, 3 = feliz, 4 = triste, 5 = nervosa, 6 = medo, 7 = nojo, 8 = surpreso
emotions_list = ['neutra', 'calma', 'feliz', 'triste', 'nervosa', 'medo', 'nojo', 'surpreso']
emotion_dict = {em[0]+1:em[1] for em in enumerate(emotions_list)}

# Visualizando os dados em um dataframe/tabela transposta (colunas viram linhas ou vice-versa --> função .T)
df = pd.DataFrame([emotion, voc_channel, modality, intensity, actors, phrase, full_path]).T

# Criando as colunas, dando nomes a elas
df.columns = ['emotion', 'voc_channel', 'modality', 'intensity', 'actors', 'phrase', 'path']

# Mapeando/Buscando cada dado a ser exibido
df['emotion'] = df['emotion'].map(emotion_dict)
df['voc_channel'] = df['voc_channel'].map({1: 'fala', 2:'musica'})
df['modality'] = df['modality'].map({1: 'AV completo', 2:'apenas video', 3:'apenas audio'})
df['intensity'] = df['intensity'].map({1: 'normal', 2:'forte'})
df['actors'] = df['actors'].apply(lambda x: 'feminino' if x%2 == 0 else 'masculino')
df['phrase'] = df['phrase'].map({1: 'Kids are talking by the door', 2:'Dogs are sitting by the door'}) # Não traduzimos a frase, pois nesta bsae de dados está sendo falada em inglês
df['path'] = df['path'].apply(lambda x: x[0] + '/' + x[1])

df.emotion.value_counts()

#%%
df.head()

#%%
'''Concatenando as bases'''
# Ajustar a bases de dados RAVDESS para ficar na mesma estrutura da base TESS
RAV_df = df.copy()

RAV_df.insert(0, "emotion_label", RAV_df.emotion, True)
RAV_df.insert(1, "source", "RAV", True)
RAV_df = RAV_df.drop(['emotion', 'voc_channel', 'modality', 'intensity', 'phrase'], 1)
RAV_df.head()

#%%
df_combined = pd.concat([TESS_df, RAV_df])
df_combined.info()

#%%
df_combined.head()
#%%
df_combined.emotion_label.value_counts()

# %%
n_files = df_combined.shape[0]

rnd = np.random.randint(0, n_files)

fname = df_combined.path[rnd]
print(f'[INFO] arquivo de audio No. {rnd}, path: {fname}')

#%%
data, sampling_rate = librosa.load(fname, sr=44100)

plt.figure(figsize=(15, 5))
info = df_combined.iloc[rnd].values
title_txt = f'emoção: {info[0]} ({info[1]})'
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

#%%
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

for path in tqdm(df_combined.path.values):
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
y = np.array(df_combined.emotion_label.tolist())
# também pode ser usado para treinar um classificador de genero, apenas alterando a classe de df.emotion para df.actors
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

#%%
labelencoder.classes_
# %%
y[4]

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

# %%
'''Criar a rede'''
model=Sequential()

# Camada de entrada
model.add(Conv1D(64, kernel_size=(5), activation='relu',input_shape=(X_train.shape[1],1)))

model.add(Conv1D(128, kernel_size=(5),activation='relu', padding='same'))
model.add(MaxPooling1D(pool_size=(5)))

model.add(Conv1D(256, kernel_size=(5),activation='relu', padding='same'))
model.add(MaxPooling1D(pool_size=(5)))
model.add(Dropout(0.2))

model.add(Flatten())

# Camadas Densas
model.add(Dense(64, activation='relu'))

# Camada de saída
# Softmax: problema de classificação de mais de 2 classes
model.add(Dense(num_labels))
model.add(Activation('softmax'))

# Compilar o modelo
# loss='categorical_crossentropy' por ser um problema com mais de 2 classes
# optimizer='adam' para fazer a atualização dos pesos
# metrics=['accuracy'] taxa de acerto
model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')
model.summary()

# %%
'''Treinando o modelo'''
num_epochs = 100
num_batch_size = 64

checkpointer = ModelCheckpoint(filepath='saved_models/emotion_recognition_5.hdf5', 
                               verbose=1, save_best_only=True)
                               
start = datetime.now()


model_history = model.fit(X_train, Y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, Y_test), callbacks=[checkpointer], verbose=1)


duration = datetime.now() - start
print("[INFO] treinamento concluído em: ", duration)

# %%
'''Avaliando o modelo'''
test_accuracy=model.evaluate(X_test,Y_test,verbose=0)
print(f"[INFO] accuracy: {test_accuracy[1]*100:.2f}%")

# https://stackoverflow.com/questions/68836551/keras-attributeerror-sequential-object-has-no-attribute-predict-classes

#%%
# As métricas são armazenadas em um dicionário no membro history do objeto retornado.
# listar as métricas coletadas
print(model_history.history.keys())
#%%
'''Exbir o hístórico de precisão e perda do modelo'''
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# plt.savefig('results/EmotionRecognition_Model_Accuracy.png')
plt.show()

plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# plt.savefig('results/EmotionRecognition_Model_Loss.png')
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

# %%
'''Testando em um audio'''
SAMPLE_AUDIO = "testes/feminino2.wav"
#%%
audio, sample_rate = librosa.load(SAMPLE_AUDIO, res_type='kaiser_fast') 
mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
mfccs_scaled_features = mfccs_scaled_features[:,:,np.newaxis]

predictions = model.predict(mfccs_scaled_features, batch_size=32)
plt.bar(labelencoder.classes_, predictions[0], color=list('rgbkymc'))

predictions=predictions.argmax(axis=1)
predictions
predictions = predictions.astype(int).flatten()
predictions = (labelencoder.inverse_transform((predictions)))

predictions

#%%
# Plotar o formato do sinal de áudio (como ele se comporta ao longo do tempo) + a classe
plt.figure(figsize=(14,5))
plt.title('Emoção: ' + str(predictions[0]).upper(), size=16)
ld.waveplot(audio, sr=sample_rate)
# %%
