#%%
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio

import librosa
import librosa.display as ld

AUDIO_SOURCE = 'testes\dog.wav'

data, sampling_rate = librosa.load(AUDIO_SOURCE)
Audio(data=data, rate=sampling_rate)

#%%
# Sample rate é o número de vezes por segundo em que as frequências são registradas
# Também o librosa usa a normalização dos sinais de áudio (-1 até 1). Por padrão qualquer áudio será convertido para esta taxa de amostragem: 22050
sampling_rate # visualizando a frequência de amostragem (Hz -> hertz)
# data # dados
# print(data.shape) # dimensão dos dados

print('[INFO] sample rate   {} Hz'.format(sampling_rate))
print('[INFO] tamanho do audio   {:3.2f} seconds'.format(len(data)/sampling_rate))
#%%
# Plotar o formato do sinal de áudio, como ele se comporta ao longo do tempo
plt.figure(figsize=(14,5))
ld.waveplot(data, sr=sampling_rate)

# Plotando o espectograma em unidade de decibéis (to_db)
DATA = librosa.stft(data)
DATAdb = librosa.amplitude_to_db(abs(DATA))
# # DATAdb = librosa.amplitude_to_db(DATA, ref = np.max)

plt.figure(figsize=(14,5))
ld.specshow(DATAdb, sr=sampling_rate, x_axis='time', y_axis='hz')
plt.colorbar(format = '% + 2.0f db')

# # Mostrando a relação entre os decibéis e as frequências em determinado período de tempo
plt.figure(figsize=(14,5))
plt.magnitude_spectrum(data, scale = 'dB')

#%%
'''Extrair os recursos'''
# Aqui, usaremos os coeficientes cepstrais de frequência de Mel (MFCC) das amostras de áudio. O MFCC resume a distribuição de frequência no tamanho da janela, portanto, é possível analisar as características de frequência e tempo do som. Essas representações de áudio nos permitirão identificar recursos para classificação

# librosa.feature.mfcc(y=None, sr=22050, S=None, n_mfcc=20, dct_type=2, norm='ortho', lifter=0, **kwargs)
mfccs = librosa.feature.mfcc(y = data, sr = sampling_rate, n_mfcc = 40)
print(mfccs.shape)

# Visualizando a série MFCC
fig, ax = plt.subplots()
img = librosa.display.specshow(mfccs, x_axis='time', ax=ax)
fig.colorbar(img, ax=ax)
ax.set(title='MFCC')

#%%
# Plotando o espectograma em unidade de decibéis (to_db)
DATAdb = librosa.amplitude_to_db(abs(mfccs))
# DATAdb = librosa.amplitude_to_db(DATA, ref = np.max)
plt.figure(figsize=(14,5))
ld.specshow(DATAdb, sr=sampling_rate, x_axis='time', y_axis='hz')
plt.colorbar(format = '% + 2.0f db')

#%%
# scipy.io.wavfile.read: retorna a taxa de amostragem (amostras/sec) e os dados de um arquivo LPCM WAV
from scipy.io import wavfile as wav
wave_sample_rate, wave_audio = wav.read(AUDIO_SOURCE)

#%%
wave_sample_rate  # visualizando a frequência de amostragem (Hz -> hertz)

#%%
# Onda sonora
# Cada um dos sinais de áudio podem ser representados por números inteiros
# A combinação destes valores formam a onda sonora
wave_audio # o formato do sinal de áudio, como ele se comporta ao longo do tempo

# %%
# Visualizando a classificação deste dataset: UrbanSound8k
# pois este é um dataset já rotulado
import pandas as pd
DATA = 'UrbanSound8K/UrbanSound8K/metadata/UrbanSound8K.csv'
metadata = pd.read_csv(DATA)
metadata.head(10)

# %%
# Verificando se o dataset está desbalanceado
metadata['class'].value_counts()