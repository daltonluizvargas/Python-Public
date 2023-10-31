#%%
'''Importando as bibliotecas'''
import os

# pathlib — Caminhos do sistema de arquivos orientado a objetos
# https://docs.python.org/pt-br/3.8/library/pathlib.html
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Definição do modelo, camadas, função de ativação,...
from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

import seaborn as sns
sns.set() # sobrescrevemos a aparência 'mat plot lib' por 'sns.set ()', para tirar vantagem do estilo nativo do seaborn.
# Em essência, isso nos permitirá codificar os gráficos em Matplotlib, mas eles serão exibidos com o que alguns chamam de “aparência Seaborn muito superior”.

#%%
# O conjunto de dados original consiste em mais de 105.000 arquivos de áudio WAV de pessoas dizendo trinta palavras diferentes. Esses dados foram coletados pelo Google e divulgados sob uma licença CC BY.
# Usaremos uma parte do conjunto de dados para economizar tempo com o carregamento de dados. Mini_Speech possue apenas 10 palavras chave: 'down' 'go' 'left' 'no' 'right' 'stop' 'up' 'yes'
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
print('[INFO] total de registros:', num_samples)
print('[INFO] registros por classe:', len(tf.io.gfile.listdir(str(DATASET_MINI_SPEECH/commands[0]))))
print('[INFO] arquivo de exemplo:', filenames[0])

#%%
# Dividir os arquivos em conjuntos de treinamento, validação e teste usando uma proporção de 80:10:10
train_files = filenames[:6400]
val_files = filenames[6400: 6400 + 800]
test_files = filenames[-800:]

print('[INFO] tamanho do conjunto de treinamento', len(train_files))
print('[INFO] tamanho do conjunto de validação', len(val_files))
print('[INFO] tamanho do conjunto de teste', len(test_files))

#%%
# Lendo os arquivos de audio e suas classes:
# O arquivo de áudio será inicialmente lido como um arquivo binário.
# tf.audio.decode_wav retorna o áudio codificado em WAV como um Tensor e a taxa de amostragem. Um arquivo WAV contém dados de série temporal com um número definido de amostras por segundo. Cada amostra representa a amplitude do sinal de áudio naquele momento específico. Em um sistema de 16 bits, como os arquivos em mini_speech_commands , os valores variam de -32768 a 32767. A taxa de amostragem para este conjunto de dados é 16kHz. Note-se que tf.audio.decode_wav vai normalizar os valores para o intervalo [-1,0, 1,0].
def decode_audio(audio_binary):
  audio, _ = tf.audio.decode_wav(audio_binary) 
  return tf.squeeze(audio, axis=-1)

# Teste da função
# audio_binary = tf.io.read_file(filenames[10])
# print(decode_audio(audio_binary))

#%%
# O atributo classe de cada arquivo WAV é seu diretório pai
def get_label(file_path):
  # Ao acessar o caminho/diretório de cada arquivo, precisamos pegar somente a classe a que ele corresponde, então usamos a função tf.strings.split passando o caminho completo até o arquivo e definindo um separador de caminhos com a função os.path.sep, ou seja, cada diretório estará separado dentro desta lista
  # Exemplo: 
  # Este é o path: Datasets\\mini_speech_commands\\yes\\8134f43f_nohash_4.wav
  # Usando um separador por diretórios do sistema (os.path.sep) temos a saída: 
  # [b'Datasets' b'mini_speech_commands' b'yes' b'8134f43f_nohash_4.wav']
  # [[-4]Datasets, [-3]mini_speech_commands, [-2]yes, [-1]8134f43f_nohash_4.wav]
  # Fazendo uma recursão nos indices de cada diretório, conseguimos retornar somente a classe de cada arquivo nos dois diretórios anteriores [-2] (diretório pai)
  parts = tf.strings.split(file_path, os.path.sep)

  return parts[-2]

# Teste da função
# print(get_label(filenames[6000]))

#%%
def get_waveform_and_label(file_path):
  label = get_label(file_path)
  audio_binary = tf.io.read_file(file_path)
  waveform = decode_audio(audio_binary)
  return waveform, label

# Teste da função
# print(get_waveform_and_label(filenames[10]))

#%%
# Otimizando o processamento através da pré-busca

# Uma etapa de treinamento inclui abrir um arquivo, buscar uma entrada de dados do arquivo e usar os dados para treinamento. Podemos ver ineficiências claras aqui, como quando nosso modelo está em treinamento, o pipeline de entrada está inativo e quando o pipeline de entrada está buscando os dados, nosso modelo está inativo.
# A pré-busca resolve as ineficiências desta abordagem, pois visa sobrepor o pré-processamento e a execução do modelo da etapa de treinamento. Em outras palavras, quando o modelo está executando a etapa de treinamento n, o pipeline de entrada estará lendo os dados para a etapa n + 1.
# tf.data executa um algoritmo de otimização para encontrar uma boa alocação de recursos de CPU e ajusta estes valores dinamicamente em tempo de execução com AUTOTUNE
# Mais detalhes aqui: https://towardsdatascience.com/optimising-your-input-pipeline-performance-with-tf-data-part-1-32e52a30cac4
AUTOTUNE = tf.data.AUTOTUNE

# tf.data.Dataset.from_tensor_slices() obtemos as fatias de um array na forma de objetos, ou seja, cada elemento da lista será retornado separadamente
files_ds = tf.data.Dataset.from_tensor_slices(train_files)
# for files in files_ds: # Demonstração do que acontece
#   print(files.numpy())
  
# Como os elementos de entrada são independentes uns dos outros, o pré-processamento/mapeamento dos elementos pode ser paralelizado em vários núcleos da CPU. Precisamos mapear a saída como a mesma estrutura dos elementos de entrada.
# #num_parallel_calls argumento para especificar o nível de paralelismo que desejado
waveform_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)

#%%
# Podemos visualizar algumas formas de onda de áudio com seus rótulos correspondentes.
# Definir o número de linhas / colunas da grade do subplot.
rows = 3
cols = 3
n = rows*cols
fig, axes = plt.subplots(rows, cols, figsize=(10, 12))

# Percorremos a lista com todos os registros por classe obtida nos processos anteriores (waveform_ds) e usamos a função take() para fazer isto somente pelo número de vezes que corresponde ao tamanho da matriz de exibição que definimos (n = 3*3 --> n = 9), assim a exibicição será de 9 gráficos ou wave forms neste caso.
# Sintaxe: numpy.take (array, indices, axis = None, out = None, mode = 'raise')
for i, (audio, label) in enumerate(waveform_ds.take(n)):
  r = i // cols # divisão inteira, arredondando para baixo. Ex.: 5 // 2 resulta em 2
  c = i % cols # resto da divisão. Ex.: 5 % 2 resulta em 1
  ax = axes[r][c] # definição dos eixos
  ax.plot(audio.numpy())
  label = label.numpy().decode('utf-8') # decodifica a string para o padrão utf-8, assim podemos visualizar somente o elemento string do Tensor
  ax.set_title(label)

plt.show()

#%%
def get_spectrogram(waveform):
  # Precisamos que as formas de onda tenham o mesmo comprimento, de modo que, ao convertê-las em uma imagem de espectrograma, os resultados tenham dimensões semelhantes. Isso pode ser feito simplesmente zerando os clipes de áudio menores que um segundo e depois concatenando ao áudio original.
  # Exemplo: se o clipe de áudio tiver 24000, então 16000-24000 = 8000, estes 8000 serão zerados
  # se o clipe de áudio tiver 10000, então 16000-10000 = 6000, estes 6000 serão zerados
  # Assim mantemos as formas de onda de entrada sempre com o mesmo comprimento de 16000
  zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)

  # Concatene o áudio com preenchimento para que todos os clipes de áudio tenham a mesma duração.
  # A função tf.cast () é usada para converter um Tensor especificado para um novo tipo de dados.
  # Esta função aceita dois parâmetros que são ilustrados abaixo:
  #   x: O tensor de entrada
  #   dtype: O tipo de dados do Tensor de saída.
  #   Ele retorna um tensor fundido de um novo tipo de dados.
  waveform = tf.cast(waveform, tf.float32)
  
  # A função tf.concat () é usada para concatenar tensores.
  # Esta função aceita três parâmetros que são ilustrados abaixo:
  #   valores: É um tensor ou lista de tensores.
  #   eixo: é o tensor 0-D que representa a dimensão a concatenar.
  #   nome (opcional): define o nome da operação.
  equal_length = tf.concat([waveform, zero_padding], 0) 

  # Converter a forma de onda em um espectrograma, que mostra as mudanças de frequência ao longo do tempo e pode ser representado como uma imagem 2D. Isso pode ser feito aplicando a transformada de Fourier de curta duração (STFT) para converter o áudio no domínio da frequência do tempo.
  # A STFT ( tf.signal.stft ) divide o sinal em janelas de tempo e executa uma transformada de Fourier em cada janela, preservando algumas as informações de tempo, e retornando um tensor 2D que você pode executar convoluções   
  # A função tf.signal.stft() é usada para calcular a Transformada de Fourier de curto prazo de sinais.
  # Usamos os parâmetros frame_length e frame_step de forma que a "imagem" do espectrograma gerada seja quase quadrada.
  # Esta função aceita cinco parâmetros:
  #   sinal: Tensor de sinais de valor real.
  #   frame_length: Eixo horizontral representa o comprimento da janela da amostra. Quanto maior o comprimento/número de amostras, mais detalhado será o espectograma, com uma resolução de frequência melhor, sendo que onde a cor no espectro é mais intensa, maior é a magnitude, porém com comprimentos de amostras maiores, a intensidade da cor diminue, ou seja, a resolução de frequência também diminue, pois as informações ficam mais dispersas. Pode fazer o teste aumentando o frame_length para 1024
  #   frame_step: Eixo vertical representa o número de amostras para a etapa. Esta é a dimensão de tempo
  #   fftLength: o tamanho da FFT a ser aplicado. Se não for fornecido, usa a menor potência de 2 sobre o frame_length.
  #   windowFn: um chamável que ocupa um comprimento de janela.
  spectrogram = tf.signal.stft(
      equal_length, frame_length=255, frame_step=128)

  # Precisamos somente da magnitude, então pegamos o valor absoluto do espectrograma usando a função tf.abs(), independente se o resultado será positivo ou nagativo, apenas será retornado o valor
  spectrogram = tf.abs(spectrogram)

  return spectrogram
#%%
for waveform, label in waveform_ds.take(1):
  label = label.numpy().decode('utf-8') # decodifica a string para o padrão utf-8, assim podemos visualizar somente o elemento string do Tensor
  spectrogram = get_spectrogram(waveform)

print('[INFO] classe:', label)
print('[INFO] dimensão do Waveform:', waveform.shape)
print('[INFO] dimensões do Espectrograma:', spectrogram.shape)
print('[INFO] áudio')
display.display(display.Audio(waveform, rate=16000))

#%%
# Função para exibir um espectrograma
def plot_spectrogram(spectrogram, ax):
  # numpy.log () é uma função matemática usada para calcular o logaritmo natural de x (x pertence a todos os elementos da matriz de entrada). É o inverso da função exponencial, bem como um logaritmo natural a nível de elemento. 
  # No exemplo abaixo, convertemos as frequências/espectrogramas para registrar a escala e transpor, de modo que o tempo seja representado no eixo x (colunas).
  # numpy.finfo() mostra os limites da máquina para tipos de ponto flutuante, ou seja, usamos isto para obter o menor número positivo possível que o tipo de dados float pode representar em minha máquina e adicionamos um epsilon (eps) para evitar log de zero.
  log_spec = np.log(spectrogram.T+np.finfo(float).eps)

  # Dimensões de escala do espectrograma
  height = log_spec.shape[0] # Altura
  width = log_spec.shape[1] # Comprimento

  # np.linspace retorna números espaçados de modo uniforme em um intervalo. Dessa forma, dado um ponto inicial e de parada, assim como a quantidade de valores, linspace irá distribuí-los uniformemente para você em uma matriz NumPy. Isso é especialmente útil para visualizações de dados e declaração de eixos durante a plotagem.
  # exemplo: 
  # np.linspace(1.0, 10.0, num = 5) # 1.0 = ponto de partida, 10.0 = ponto de parada, 5 = quantidade de valores
  # >>> array([ 1. , 3.25, 5.5 , 7.75, 10. ])
  X = np.linspace(0, np.size(spectrogram), num=width, dtype=int) # 0 = ponto de partida, np.size(spectrogram) = ponto de parada, num = width número de vezes que serão distribuídos, dtype = int e que serão do tipo inteiro

  # Função range é usada para criar o eixo Y na mesma altura do espectrograma
  Y = range(height)
  ax.pcolormesh(X, Y, log_spec)


fig, axes = plt.subplots(2, figsize=(12, 8))
timescale = np.arange(waveform.shape[0])
axes[0].plot(timescale, waveform.numpy())
axes[0].set_title('Waveform')
axes[0].set_xlim([0, 16000])

plot_spectrogram(spectrogram.numpy(), axes[1])
axes[1].set_title('Spectrogram')
plt.show()

#%%
def get_spectrogram_and_label_id(audio, label):
  spectrogram = get_spectrogram(audio)

  # A função tf.expand_dims é semelhante ao reshape, alterando a forma de um tensor adicionando dimensões
  spectrogram = tf.expand_dims(spectrogram, -1)

  # argmax () é um método presente no módulo matemático tensorflow. Este método é usado para encontrar o índice com o maior valor no eixo de um tensor.
  label_id = tf.argmax(label == commands)

  return spectrogram, label_id

#%%
# Novamente otimizamos o processamento através da pré-busca
spectrogram_ds = waveform_ds.map(
    get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)

#%%
# Podemos visualizar algumas formas de onda de áudio com seus rótulos correspondentes.
# Definir o número de linhas / colunas da grade do subplot.
rows = 3
cols = 3
n = rows*cols
fig, axes = plt.subplots(rows, cols, figsize=(10, 10))

# Percorremos a lista com todos os registros por classe obtida nos processos anteriores (spectrogram_ds) e usamos a função take() para fazer isto somente pelo número de vezes que corresponde ao tamanho da matriz de exibição que definimos (n = 3*3 --> n = 9), assim a exibicição será de 9 gráficos ou wave forms neste caso.
# Sintaxe: numpy.take (array, indices, axis = None, out = None, mode = 'raise')
for i, (spectrogram, label_id) in enumerate(spectrogram_ds.take(n)):
  r = i // cols # divisão inteira, arredondando para baixo. Ex.: 5 // 2 resulta em 2
  c = i % cols # resto da divisão. Ex.: 5 % 2 resulta em 1
  ax = axes[r][c] # definição dos eixos

  # A função numpy.squeeze() é usada quando queremos remover as dimensões de tamanho 1 do ndarray. Também especificamos o índice da dimensão a ser removida no segundo argumento axis de numpy.squeeze(). As dimensões que não são o índice especificado não são removidas.
  # https://note.nkmk.me/en/python-numpy-squeeze/
  plot_spectrogram(np.squeeze(spectrogram.numpy()), ax)
  
  ax.set_title(commands[label_id.numpy()])
  ax.axis('off')

plt.show()

#%%
'''Repita o pré-processamento do conjunto de treinamento nos conjuntos de validação e teste'''
def preprocess_dataset(files):
  # tf.data.Dataset.from_tensor_slices() obtemos as fatias de um array na forma de objetos, ou seja, cada elemento da lista será retornado separadamente
  files_ds = tf.data.Dataset.from_tensor_slices(files)

  # Novamente otimizamos o processamento através da pré-busca
  output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
  output_ds = output_ds.map(
      get_spectrogram_and_label_id,  num_parallel_calls=AUTOTUNE)

  return output_ds

#%%
train_ds = spectrogram_ds

# Passamos como parâmetro os arquivos de validação e teste
val_ds = preprocess_dataset(val_files) 
test_ds = preprocess_dataset(test_files)

#%%
batch_size = 64

# dataset.batch combina elementos consecutivos deste conjunto de dados em lotes, em outras palavras, pegará as primeiras 64 entradas e fará um lote com elas e assim por diante com os demais registros criando lotes com o tamanho 64.
# https://www.gcptutorials.com/article/how-to-use-batch-method-in-tensorflow
train_ds = train_ds.batch(batch_size)
val_ds = val_ds.batch(batch_size)

#%%
# dataset.cache() e dataset.prefetch() operações para reduzir a latência de leitura durante o treinamento do modelo
# dataset.cache() pode armazenar em cache um conjunto de dados, seja na memória ou no armazenamento local. Isso evitará que algumas operações (como abertura de arquivo e leitura de dados) sejam executadas durante cada época. As próximas épocas reutilizarão os dados armazenados em cache pela transformação do cache.
# É uma ótima maneira de melhorar a eficiência da consulta, bem como minimizar os custos de computação.
# Ao armazenar em cache, os dados armazenados em cache persistirão nas execuções. Mesmo a primeira iteração através dos dados será lida do arquivo de cache. Alterar o pipeline de entrada antes da chamada para .cache()não terá efeito até que o arquivo de cache seja removido ou o nome do arquivo seja alterado.
# prefetch() novamente otimizamos o processamento através da pré-busca, porém desta vez sem realizar a transformação dos dados, apenas sobrepondo o pré-processamento e a execução do modelo de uma etapa de treinamento. Enquanto o modelo está executando as etapas de treinamento s, o pipeline de entrada está lendo os dados para a etapa s + 1. Isso reduz o tempo da etapa ao máximo (em oposição à soma) do treinamento e o tempo que leva para extrair os dados.
train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

#%%
# Percorremos a lista de espectrogramas e usamos a função take() para fazer isto somente por 1 vez a fim de obter o formato de entrada/dimensões dos dados
# Sintaxe: numpy.take (array, indices, axis = None, out = None, mode = 'raise')
for spectrogram, _ in spectrogram_ds.take(1):
  input_shape = spectrogram.shape
print('[INFO] dimensões:', input_shape)
num_labels = len(commands)

#%%
# layers.Normalization() adicionamos esta camada de pré-processamento para normalizar cada pixel da imagem com base na sua média e desvio padrão (entre 0 e 1)
norm_layer = layers.Normalization()

#%%
# norm_layer.adapt irá calcular a média e a variância dos dados e armazená-los como pesos da camada.
# Também aplicamos a tranformação dos dados com a função .map()
# .map() retorna um iterador, quando você itera sobre ele, o lambda será chamado em cada elemento da lista, um de cada vez, e retornará o resultado.
# Por exemplo, o mapa pode ser usado para adicionar 1 a cada elemento ou projetar um subconjunto de componentes do elemento:
dataset = tf.data.Dataset.range(1, 6)  # ==> [ 1, 2, 3, 4, 5 ]
dataset = dataset.map(lambda x: x + 1)
list(dataset.as_numpy_iterator())

#%%
# norm_layer.adapt(spectrogram_ds.map(lambda x, _: x))
norm_layer.adapt(data=spectrogram_ds.map(map_func=lambda spec, label: spec))

#%%
#https://learnopencv.com/image-classification-using-convolutional-neural-networks-in-keras/
for spectrogram, _ in spectrogram_ds.take(1):
  input_shape = spectrogram.shape
print('Input shape:', input_shape)
num_labels = len(commands)

# Instantiate the `tf.keras.layers.Normalization` layer.
norm_layer = layers.Normalization()
# Fit the state of the layer to the spectrograms
# with `Normalization.adapt`.
norm_layer.adapt(data=spectrogram_ds.map(map_func=lambda spec, label: spec))

model = models.Sequential([
    layers.Input(shape=input_shape),
    # Downsample the input.
    layers.Resizing(32, 32),
    # Normalize.
    norm_layer,
    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(64, 3, activation='relu'),
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
    optimizer=tf.keras.optimizers.Adam(),
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
    train_ds, 
    validation_data=val_ds,  
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
'''Avalie o desempenho do modelo'''
# Execute o modelo no conjunto de teste e verifique o desempenho do modelo
test_audio = []
test_labels = []

for audio, label in test_ds:
  test_audio.append(audio.numpy())
  test_labels.append(label.numpy())

test_audio = np.array(test_audio)
test_labels = np.array(test_labels)

#%%
y_pred = np.argmax(model.predict(test_audio), axis=1)
y_true = test_labels

test_acc = sum(y_pred == y_true) / len(y_true)
print(f'[INFO] precisão do conjunto de teste: {test_acc:.0%}')

#%%
'''Exibir uma matriz de confusão'''
# Use uma matriz de confusão para verificar quão bem o modelo classifica cada um dos comandos do conjunto de teste
confusion_mtx = tf.math.confusion_matrix(y_true, y_pred) 
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx, xticklabels=commands, yticklabels=commands, 
            annot=True, fmt='g', cmap='coolwarm')
plt.xlabel('Classes Previstas')
plt.ylabel('Calsses Reais')
plt.show()

#%%
'''Executar inferência em um arquivo de áudio'''
# Finalmente, verifique a saída de previsão do modelo usando um arquivo de áudio de entrada de alguém dizendo "não"
sample_file = data_dir/'no/01bb6a2a_nohash_0.wav'

sample_ds = preprocess_dataset([str(sample_file)])

for spectrogram, label in sample_ds.batch(1):
  prediction = model(spectrogram)
  plt.bar(commands, tf.nn.softmax(prediction[0]), color=list('rgbkymc'))
  plt.title(f'Previsões para o comando "{commands[label[0]]}"')
  plt.show()

# %%
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
