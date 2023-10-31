#%%
# Importando as bibliotecas
import pandas as pd
import os
import numpy as np

import matplotlib.pyplot as plt
from IPython.display import Audio

import librosa
import librosa.display as ld

# tqdm é uma biblioteca em Python usada para criar medidores de progresso ou barras de progresso. O nome tqdm vem do nome árabe taqaddum, que significa 'progresso'
from tqdm import tqdm


#%%
# Carregando o dataset
DATASET_PATH = 'Datasets/Audio_Speech_Actors/'
# DATASET_PATH = 'Datasets/Audio_Song_Actors/'

#%%
# Identificando os atributos/características em cada arquivo de áudio da base de dados RAVDESS
'''
Modalidade (01 = AV completo, 02 = apenas vídeo, 03 = apenas áudio).
Canal vocal (01 = fala, 02 = música).
Emoção (01 = neutro, 02 = calmao, 03 = feliz, 04 = triste, 05 = zangado, 06 = com medo, 07 = nojo, 08 = surpreso).
Intensidade emocional (01 = normal, 02 = forte). NOTA: Não há intensidade forte para a emoção 'neutra'.
Frase (01 = "Crianças conversam perto da porta", 02 = "Cachorros estão sentados na porta").
Repetição (01 = 1ª repetição, 02 = 2ª repetição).
Ator (01 a 24. Os atores com números ímpares são homens, os atores com números pares são mulheres).

Exemplo Audio_musica_Actors/Actor_01/03-02-01-01-01-01-01.wav:
Modalidade 03: Apenas áudio
Canal vocal 02: música
Emoção 01: neutro
Intensidade emocional 01: normal. NOTA: Não há intensidade forte para a emoção 'neutra'.
Frase 01: "Crianças conversam perto da porta"
Repetição 01: 1ª repetição
Ator 01: 1º ator - homem, já que o número de identificação do ator é impar'''

modality = [] # Modalidade (01 = AV completo, 02 = apenas vídeo, 03 = apenas áudio).
voc_channel = [] # Canal vocal (01 = fala, 02 = música).
emotion = [] # Emoção (01 = neutro, 02 = calma, 03 = feliz, 04 = triste, 05 = zangado, 06 = com medo, 07 = nojo, 08 = surpreso).
intensity = [] # Intensidade emocional (01 = normal, 02 = forte). NOTA: Não há intensidade forte para a emoção 'neutra'.
phrase =[] # Frase (01 = "Crianças conversam perto da porta", 02 = "Cachorros estão sentados na porta").
actors = [] # Ator (01 a 24. Os atores com números ímpares são homens, os atores com números pares são mulheres)

full_path = []

for root, dirs, files in tqdm(os.walk(DATASET_PATH)):
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

#%%
# Criamos um dicionário de emoções (8), ajutando-a para iniciar a partir do índice 1 ao 8, ficando desta maneira:
# 1 = neutra, 2 = calma, 3 = feliz, 4 = triste, 5 = nervosa, 6 = medo, 7 = nojo, 8 = surpreso
emotions_list = ['neutra', 'calma', 'feliz', 'triste', 'nervosa', 'medo', 'nojo', 'surpreso']
emotion_dict = {em[0]+1:em[1] for em in enumerate(emotions_list)}

emotion_dict
#%%
# Visualizando os dados em um dataframe/tabela transposta (colunas viram linhas ou vice-versa --> função .T)
df = pd.DataFrame([emotion, voc_channel, modality, intensity, actors, phrase, full_path]).T

#%%
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

df.head()


#%%
df.shape[0]

#%%
df.emotion.value_counts()

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
title_txt = f'voz: {info[4]} - emoção: {info[0]} ({info[1]}, {info[3]})'
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
# Salvando as informações de classificação da base de dados, assim temos as informações detalhadas de cada arquivo de audio
df.to_csv(os.path.join(DATASET_PATH,"metadata.csv"), index=False)

# %%
DATA = DATASET_PATH + '/metadata.csv'
DATA

#%%
'''Extraindo recursos/características MFCC's de cada arquivo de áudio do dataset'''
metadata = pd.read_csv(DATA)
metadata.head()

#%%
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

for path in tqdm(df.path.values):
  data = features_extractor(path)
  extracted_features.append([data])

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
y = np.array(df.emotion.tolist())
# também pode ser usado para treinar um classificador de genero, apenas alterando a classe de df.emotion para df.actors
# %%
X
# %%
y
# %%
#%%
'''Criando o modelo'''
import tensorflow as tf
print(tf.__version__)

# %%
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Transformando os valores categórios em números utilizando o LabelEncoder
# Assim para cada registro já calssificado, os codifica para um número entre 0 e 1
labelencoder = LabelEncoder()
y = to_categorical(labelencoder.fit_transform(y))
# %%
y[0]

# %%
# Separando a base de dados em treinamento e teste
from sklearn.model_selection import train_test_split

# Usamos 20% (0.2) para testar e 80% para treinar
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# %%
X_train

#%%
plt.imshow(X_train[10, :, :, 0])

# %%
y

#%%
X_train.shape

# %%
X_test.shape

# %%
Y_train.shape

# %%
Y_test.shape

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn import metrics

# %%
# Número de classes
num_labels = y.shape[1]

# %%
num_labels

# %%
'''Criar a rede'''
model=Sequential()
###first layer
model.add(Dense(512,input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.1))
###second layer
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.1))
###third layer
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.1))

###final layer
model.add(Dense(num_labels))
model.add(Activation('softmax'))

# %%
model.summary()

# %%
# otimizador = tf.keras.optimizers.Adam(learning_rate=0.0006)
model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')

# %%
'''Treinando o modelo'''

## Trianing my model
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime 

num_epochs = 1000
num_batch_size = 32

checkpointer = ModelCheckpoint(filepath='saved_models/emotion_recognition.hdf5', 
                               verbose=1, save_best_only=True)
start = datetime.now()

model.fit(X_train, Y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, Y_test), callbacks=[checkpointer], verbose=1)


duration = datetime.now() - start
print("Training completed in time: ", duration)

# %%
'''Avaliando o modelo'''
test_accuracy=model.evaluate(X_test,Y_test,verbose=0)
print(test_accuracy[1])

# https://stackoverflow.com/questions/68836551/keras-attributeerror-sequential-object-has-no-attribute-predict-classes

# %%
'''Testando em um audio'''
AUDIO="testes/triste.wav"
audio, sample_rate = librosa.load(AUDIO, res_type='kaiser_fast') 
mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)

predict_x=model.predict(mfccs_scaled_features)
classes_x=np.argmax(predict_x,axis=1)

prediction_class = labelencoder.inverse_transform(classes_x)
prediction_class

#%%
# Plotar o formato do sinal de áudio, como ele se comporta ao longo do tempo
plt.figure(figsize=(14,5))
plt.title('Classe: ' + str(prediction_class[0]).upper(), size=16)
ld.waveplot(data, sr=sample_rate)
# %%
