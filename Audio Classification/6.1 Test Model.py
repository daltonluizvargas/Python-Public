# %%
import tensorflow as tf
import numpy as np
import pandas as pd

import librosa
import librosa.display as ld

import matplotlib.pyplot as plt

import seaborn as sns
sns.set()

#%%
MODEL_TYPES = ['GÊNERO', 'EMOÇÃO', 'COMANDO', 'SOM']

#%%
MODEL_TYPES[3]

#%%
def load_model_by_name(model_type):
    if model_type == MODEL_TYPES[0]:
        model = tf.keras.models.load_model('saved_models/genre_recognition.hdf5')
        model_dict = sorted(list(['feminino', 'masculino']))
    if model_type == MODEL_TYPES[1]:
        model = tf.keras.models.load_model('saved_models/emotion_recognition.hdf5')
        model_dict = sorted(list(['neutra', 'calma', 'feliz', 'triste', 'nervosa', 'medo', 'nojo', 'surpreso']))
    if model_type == MODEL_TYPES[2]:
        model = tf.keras.models.load_model('saved_models/voice_command_recognition.hdf5')
        model_dict = sorted(list(['down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes']))
    if model_type == MODEL_TYPES[3]:
        model = tf.keras.models.load_model('saved_models/audio_classification.hdf5')
        model_dict = sorted(list(['ar_condicionado', 'buzina_de_carro', 'crianca_brincando', 'latido_de_cachorro', 'perfuracao', 'motor_em_marcha_lenta', 'tiro_de_arma', 'britadeira', 'sirene', 'musica_de_rua']))
    
    return model, model_dict
    

#%%
model_type = 'SOM'
loaded_model = load_model_by_name(model_type)

#%%
loaded_model[0] # Modelo
#%%
loaded_model[1] # Dicinário/Classes
#%%
loaded_model[0].summary() # Estrutura do modelo

#%%
loaded_model[0].get_config()

#%%
loaded_model[0].name

# %%
'''Testando em um audio'''
SAMPLE_AUDIO = "testes/children_playing.wav"
audio, sample_rate = librosa.load(SAMPLE_AUDIO, res_type='kaiser_fast') 
mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
mfccs_scaled_features = mfccs_scaled_features[:,:,np.newaxis]


predictions = loaded_model[0].predict(mfccs_scaled_features, batch_size=32)
plt.bar(loaded_model[1], predictions[0], color=list('rgbkymc'))

predictions=predictions.argmax(axis=1)
predictions = predictions.astype(int).flatten()
predictions = loaded_model[1][predictions[0]]

# Plotar o formato do sinal de áudio (como ele se comporta ao longo do tempo) + a classe
result_str = 'Classificação de ' + model_type + ': ' + str(predictions).upper()
plt.figure(figsize=(14,5))
plt.title(result_str, size=16)
ld.waveplot(audio, sr=sample_rate)
# %%
