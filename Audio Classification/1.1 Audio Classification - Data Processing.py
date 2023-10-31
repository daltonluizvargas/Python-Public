#%%
# pip install --user -r /path/to/requirements.txt
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
# Carregando os arquivos
AUDIO_DATASET_PATH = 'UrbanSound8K/UrbanSound8K/audio/'
DATA = 'UrbanSound8K/UrbanSound8K/metadata/UrbanSound8K.csv'

# %%
'''Extraindo recursos/características MFCC's de cada arquivo de áudio do dataset'''
metadata = pd.read_csv(DATA)
metadata.head()

# Função extratora de recursos
def features_extractor(file):
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
for index_num,row in tqdm(metadata.iterrows()):
    file_name = os.path.join(os.path.abspath(AUDIO_DATASET_PATH),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
    final_class_labels=row["class"]
    data=features_extractor(file_name)
    extracted_features.append([data,final_class_labels])

# %%
extracted_features

# %%
# Convertendo os recursos extraídos para visualização com Pandas
extracted_features_df = pd.DataFrame(extracted_features, columns = ['feature', 'class'])
# extracted_features_df.shape
extracted_features_df.head()


# %%
'''Dividindo entre atributos classe(class) e atributos previsores(features)'''
X = np.array(extracted_features_df['feature'].tolist())
Y = np.array(extracted_features_df['class'].tolist())

# %%
X.shape

# %%
Y

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
Y = to_categorical(labelencoder.fit_transform(Y))
# %%
Y

# %%
# Separando a base de dados em treinamento e teste
from sklearn.model_selection import train_test_split

# Usamos 20% (0.2) para testar e 80% para treinar
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
# %%
X_train

# %%
Y

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
num_labels = Y.shape[1]

# %%
num_labels

# %%
'''Criar a rede'''
# Número de camadas ocultas
# Neurônios = (Entradas + Saídas) / 2

# Entradas são os atributos previsores da base de dados
# Pesos: No treinamento de redes neurais o objeto é encontrar os pesos para cada um dos atributos
# https://iaexpert.academy/2020/05/25/funcoes-de-ativacao-definicao-caracteristicas-e-quando-usar-cada-uma/
model=Sequential()
###first layer
model.add(Dense(100,input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###second layer
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###third layer
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))

###final layer
model.add(Dense(num_labels))
model.add(Activation('softmax'))

# %%
model.summary()

# %%
model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')

# %%
'''Treinando o modelo'''

## Trianing my model
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime 

num_epochs = 100
num_batch_size = 32

checkpointer = ModelCheckpoint(filepath='saved_models/audio_classification.hdf5', 
                               verbose=1, save_best_only=True)
start = datetime.now()

model.fit(X_train, Y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, Y_test), callbacks=[checkpointer], verbose=1)


duration = datetime.now() - start
print("Training completed in time: ", duration)

# %%
'''Avaliando o modelo'''
test_accuracy=model.evaluate(X_test,Y_test,verbose=0)
print(test_accuracy[1])

#%%
filename="dog.wav"
prediction_feature=features_extractor(filename)
prediction_feature=prediction_feature.reshape(1,-1)
# model.predict(prediction_feature)
(model.predict(prediction_feature) > 0.5).astype("int32")

# https://stackoverflow.com/questions/68836551/keras-attributeerror-sequential-object-has-no-attribute-predict-classes

#%%
X_test[1]

#%%
# predict_x = model.predict(X_test)
# classes_x=np.argmax(predict_x,axis=1)
# classes_x=np.argmax(predict_x)

(model.predict(X_test) > 0.5).astype("int32")

# %%
'''Testando em um audio'''
filename="buzina.wav"
audio, sample_rate = librosa.load(filename, res_type='kaiser_fast') 
mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)

print(mfccs_scaled_features)
mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
print(mfccs_scaled_features)
print(mfccs_scaled_features.shape)

# %%
# predicted_label=model.predict(mfccs_scaled_features)
predicted_label = (model.predict(mfccs_scaled_features) > 0.5).astype("int32")
print(predicted_label)
np.argmax(predicted_label)

# prediction_class = labelencoder.inverse_transform(predicted_label) 

# %%
prediction_class
# %%
