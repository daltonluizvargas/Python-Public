#%%
'''YAMNet (Yet Another Mobile Network)'''

'''Importando as bibliotecas'''
import seaborn as sns

sns.set()

import csv

import librosa
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from IPython.display import Audio

#%%
# Carregando o modelo que prevê 521 eventos de áudio
# https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/yamnet_class_map.csv
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

# Caso ocorro o seguinte erro:
# OSError: SavedModel file does not exist at: C:\Users\dalto\AppData\Local\Temp\tfhub_modules\9616fd04ec2360621642ef9455b84f4b668e219e\{saved_model.pbtxt|saved_model.pb}
# Então vai até a pasta especificado no erro e exclue o caminho/pasta

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

#%%
class_names

#%%
# Exibir a quantidade de classes no modelo
len(class_names)

#%%
# Exibir as 20 primeiras classes dos modelo
for name in class_names[:20]:
  print(name)
print('...')

#%%
'''Preparando o arquivo de som'''
# o tamanho mínimo do áudio é 0,975s ou 15.600 samples (já que temos uma taxa de amostragem igual a 16.000) e um tamanho de deslocamento de 0,48s
def preprocess_audio(audio):
  wav_file_name = audio
  wav_data, sample_rate = librosa.load(wav_file_name, sr=16000)

  # Mostre algumas informações básicas sobre o áudio.
  duration = len(wav_data)/sample_rate
  print(f'Sample rate: {sample_rate} Hz')
  print(f'Total duration: {duration:.2f}s')
  print(f'Size of the input: {len(wav_data)}')

  # Ouvindo o arquivo wav.
  Audio(data=wav_data, rate=sample_rate)

  #%%
  # Os resultados do modelo são:
  # scores: previsões - pontuações para cada uma das 521 classes;
  # embeddings - recursos YAMNet;
  # spectrogram - espectrogramas de patches..

  # Execute o modelo, verifique a saída.
  scores, embeddings, spectrogram = yamnet_model(wav_data)
  
  # YAMNet também retorna algumas informações adicionais que podemos usar para  visualização. Vamos dar uma olhada na forma de onda, espectrograma e as  principais classes inferidas.
  scores_np = scores.numpy()
  spectrogram_np = spectrogram.numpy()
  infered_class = class_names[scores_np.mean(axis=0).argmax()]
  print(f'O som principal é: {infered_class}')

  #%%
  plt.figure(figsize=(10, 6))

  # Plot da wav_data
  plt.subplot(3, 1, 1)
  plt.plot(wav_data)
  plt.xlim([0, len(wav_data)])

  # Plot do espectrograma log-mel retornado pelo modelo.
  plt.subplot(3, 1, 2)
  plt.imshow(spectrogram_np.T, aspect='auto', interpolation='nearest',  origin='lower')

  # Plotar e rotular as pontuações de saída do modelo para as classes de melhor pontuação.
  mean_scores = np.mean(scores, axis=0)
  top_n = 10
  top_class_indices = np.argsort(mean_scores)[::-1][:top_n]
  plt.subplot(3, 1, 3)
  plt.imshow(scores_np[:, top_class_indices].T, aspect='auto',  interpolation='nearest', cmap='gray_r')

  # patch_padding = (PATCH_WINDOW_SECONDS / 2) / PATCH_HOP_SECONDS
  # valores da documentação do modelo
  patch_padding = (0.025 / 2) / 0.01
  plt.xlim([-patch_padding-0.5, scores.shape[0] + patch_padding-0.5])
  # Rotule as classes top_N (10), ou seja, as 10 classes com a melhor pontuação   de classificação
  yticks = range(0, top_n, 1)
  plt.yticks(yticks, [class_names[top_class_indices[x]] for x in yticks])
  _ = plt.ylim(-0.5 + np.array([top_n, 0]))

#%%
sample_audio = 'testes/children_playing.wav'
preprocess_audio(sample_audio)

