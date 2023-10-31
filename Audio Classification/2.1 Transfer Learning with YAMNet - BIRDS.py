#%%
import seaborn as sns
from tensorflow.python.keras import optimizers
sns.set()

import itertools
import glob
import random
import os
import csv

import librosa

from IPython.display import Audio, Image, display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub

# A biblioteca TFLite Model Maker simplifica o processo de adaptação e conversão de um modelo de rede neural TensorFlow para dados de entrada específicos ao implantar este modelo para aplicativos de ML no dispositivo.
# pip install tflite-model-maker
import tflite_model_maker as mm
from tflite_model_maker import audio_classifier

# Comece por instalar TensorFlow I/O, o que tornará mais fácil para você para carregar arquivos de áudio off disco.
# pip install tensorflow_io
import tensorflow_io as tfio

# %%
# Relembrando sobre o YAMNet:

# YAMNet é uma rede neural pré-treinada que emprega a arquitetura MobileNetV1. Ele pode usar uma forma de onda de áudio como entrada e fazer previsões independentes para cada um dos 521 eventos de áudio do modelo.

# Internamente, o modelo extrai "quadros" do sinal de áudio e processa lotes desses quadros. Esta versão do modelo usa quadros com 0,96 segundo de duração e extrai um quadro a cada 0,48 segundos.

# O modelo aceita uma matriz 1-D float32(Tensor ou NumPy) contendo uma forma de onda de comprimento arbitrário, como representado de canal único (mono) com amostras de 16 kHz no intervalo [-1.0, +1.0]. 

# O modelo retorna 3 saídas, incluindo as notas de classe, embeddings (que você vai usar para a aprendizagem de transferência), e o log mel espectrograma. 

# Um uso específico do YAMNet é como um extrator de recursos de alto nível alimentando apenas uma comada oculta densa (tf.keras.layers.Dense) com a dimensão de 1024 neurônios. Ou seja,iremos usar recursos de entrada do modelo de base (YAMNet) e alimentá-los de forma mais rasa em nosso modelo. Em outras palavras, treinamos a rede em uma pequena quantidade de dados para a classificação de áudio sem a necessidade de uma grande quantidade de dados rotulados.

#%%
# Carregando o modelo que prevê 521 eventos de áudio
# https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/yamnet_class_map.csv
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

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
# Neste exemplo usares o conjunto de dados Birds, que é uma coleção educacional com 5 tipos de cantos de pássaros:
# White-breasted Wood-Wren (uirapuru-de-peito-branco)
# House Sparrow (Pardal)
# Red Crossbill (Cruza-bico)
# Chestnut-crowned Antpitta (Grallaria ruficapilla)
# Azara's Spinetail (espineta de Azara)
# O áudio original veio do Xeno-canto, que é um site dedicado a compartilhar sons de pássaros de todo o mundo.

# Os áudios já estão divididos em pastas de teste e treinamento. Dentro de cada pasta, há uma pasta para cada pássaro, usando seu bird_code como nome.

# Os áudios são todos mono e com taxa de amostragem de 16kHz.

# Para obter mais informações sobre cada arquivo, você pode ler o arquivo metadata.csv. Ele contém todos os autores dos arquivos, lincensas e mais algumas informações. Você não precisará ler neste tutorial.
DATASET_PATH = 'Datasets/small_birds_dataset/'
DATA = 'Datasets/small_birds_dataset/metadata.csv'

#%%
metadata = pd.read_csv(DATA)
metadata.head()

#%%
birds = np.array(tf.io.gfile.listdir(str(DATASET_PATH + '/train')))
print(f'[INFO] birds: ', birds)

#%%
bird_code_to_name = {
  'wbwwre1': 'White-breasted Wood-Wren',
  'houspa': 'House Sparrow',
  'redcro': 'Red Crossbill',  
  'chcant2': 'Chestnut-crowned Antpitta',
  'azaspi1': "Azara's Spinetail",   
}

birds_images = {
  'wbwwre1': DATASET_PATH + '\images\White-breasted Wood-Wren.jpg', # 	Alejandro Bayer Tamayo from Armenia, Colombia 
  'houspa': DATASET_PATH + '\images\House Sparrow.jpg', # 	Diliff
  'redcro': DATASET_PATH + '\images\Red Crossbill.jpg', #  Elaine R. Wilson, www.naturespicsonline.com
  'chcant2': DATASET_PATH + '\images\Chestnut-crowned Antpitta.jpg', # 	Mike's Birds from Riverside, CA, US
  'azaspi1': DATASET_PATH + "\images\Azara's Spinetail.jpg", # https://www.inaturalist.org/photos/76608368
}

#%%
for index in range(len(birds)):
    print(f'Nome do pássaro: {bird_code_to_name[birds[index]]}')
    display(Image(birds_images[birds[index]]))

#%%
test_files = os.path.join(DATASET_PATH, "test/*/*.wav")

#%%
# Função para selecionar arquivos de áudio aleatórios da base de dados
def get_random_audio_file():
  test_list = glob.glob(test_files)
  random_audio_path = random.choice(test_list)
  return random_audio_path

# Função para carregar arquivos de áudio e a imagem correspondente baseando-se no path do arquivo (contendo o código/id do pássaro)
def get_bird_data(audio_path, plot = True):
  wav_data, sample_rate = librosa.load(audio_path, sr=16000)

  bird_code = audio_path.split('\\')[-2]
  print(f'Nome do pássaro: {bird_code_to_name[bird_code]}')
  print(f'Código do pássaro: {bird_code}')

  if plot:
    display(Image(birds_images[bird_code]))

    plttitle = f'{bird_code_to_name[bird_code]} ({bird_code})'
    plt.title(plttitle)
    plt.plot(wav_data)
    display(Audio(wav_data, rate=sample_rate))

  return wav_data

print('[INFO] funções e estruturas de dados criadas')

#%%
random_audio = get_random_audio_file()
get_bird_data(random_audio)

#%%
# Os resultados do modelo são:
  # scores: previsões - pontuações para cada uma das 521 classes;
  # embeddings - recursos YAMNet;
  # spectrogram - espectrogramas
random_audio = get_random_audio_file()
scores, embeddings, spectrogram = yamnet_model(get_bird_data(random_audio))
class_scores = tf.reduce_mean(scores, axis=0)
top_class = tf.argmax(class_scores)
infered_class = class_names[top_class]

print(f'[INFO] o som principal é: {infered_class}')
print(f'[INF] a forma dos embeddings: {embeddings.shape}')

#%%
'''Treinando o modelo'''
# https://www.tensorflow.org/lite/api_docs/python/tflite_model_maker/audio_classifier
# A biblioteca do Model Maker usa a aprendizagem por transferência para simplificar o processo de treinamento de um modelo TensorFlow Lite usando um conjunto de dados personalizado. Retreinar um modelo TensorFlow Lite com seu próprio conjunto de dados personalizado reduz a quantidade de dados de treinamento e o tempo necessário.

# Model Marker fornece pontuações de classe em nível de quadro (ou seja, 521 pontuações para cada quadro). Para determinar as previsões no nível do clipe, as pontuações podem ser agregadas por classe em todos os quadros (por exemplo, usando agregação média ou máxima). Isto é feito por abaixo scores_np.mean(axis=0) . Finalmente, para encontrar a classe com melhor pontuação no nível do clipe, você pega o máximo das 521 pontuações agregadas.

# Ao usar Model Marker para classificação de áudio, você deve começar com uma especificação do modelo. Este é o modelo básico do qual seu novo modelo extrairá informações para aprender sobre as novas classes. Também afeta como o conjunto de dados será transformado para respeitar os parâmetros de especificação do modelo, como: taxa de amostragem, número de canais.

# YAMNet é um classificador de eventos de áudio treinado no conjunto de dados AudioSet para prever eventos de áudio da ontologia AudioSet.

# Espera-se que sua entrada seja de 16kHz e com 1 canal.

# Você não precisa fazer nenhuma reamostragem. O modelo YAMNet cuida disso para você:

#  * frame_length é para decidir quanto tempo terá cada amostra de treinamento (o número de amostras em cada quadro/fatia/janela de áudio). Se o arquivo de áudio for menor que frame_length, então o arquivo de áudio será ignorado. Neste caso, para garantir que isto não ocorra, podemos aplicar a seguinte fórmula: COMPRIMENTO_ESPERADO_DA_FORMA_DE_ONDA * 3 segundos, adicionado mais 3 segundos de silêncio ao áudio.
# * frame_steps é número de amostras entre dois quadros de áudio. Este valor deve ser maior que frame_length. Isto é usado para decidir a que distância estão as amostras de treinamento. Nesse caso, a iª amostra começará em COMPRIMENTO_ESPERADO_DA_FORMA_DE_ONDA * 6s após a (i-1)ª amostra.

# O motivo para definir esses valores é contornar algumas limitações no conjunto de dados do mundo real.

# Por exemplo, no conjunto de dados de pássaros, os pássaros não cantam o tempo todo. Eles cantam, descansam e cantam novamente, com ruídos intermediários. Ter um quadro longo ajudaria a capturar o canto, mas defini-lo muito longo reduzirá o número de amostras para treinamento.

spec = audio_classifier.YamNetSpec(
    keep_yamnet_and_custom_heads=True,
    frame_step=3 * audio_classifier.YamNetSpec.EXPECTED_WAVEFORM_LENGTH,
    frame_length=6 * audio_classifier.YamNetSpec.EXPECTED_WAVEFORM_LENGTH)

#%%
'''Carregando os dados'''
# O Model Maker tem a API para carregar os dados de uma pasta e tê-los no formato esperado para a especificação do modelo.

# A divisão de treinamento e teste são baseados nas pastas. O conjunto de dados de validação será criado como 20% da divisão do treinamento.

# Nota: O parâmetro cache = True é importante para tornar o treinamento mais rápido, mas também exigirá mais RAM para armazenar os dados. Para o conjunto de dados de pássaros, isso não é um problema, pois tem apenas 300 MB, mas se você usar seus próprios dados, deve prestar atenção a eles.

train_data = audio_classifier.DataLoader.from_folder(
    spec, os.path.join(DATASET_PATH, 'train'), cache=True)
train_data, validation_data = train_data.split(0.8)
test_data = audio_classifier.DataLoader.from_folder(
    spec, os.path.join(DATASET_PATH, 'test'), cache=True)

#%%
'''Executando a inferência'''
# o audio_classifier possui o método create que cria um modelo e já começa a treiná-lo.

# Podemos personalizar muitos parâmetros, para obter mais informações. Para mais detalhes favor consultar a documentação.

# Nesta primeira tentativa, Usaremos todas as configurações padrões e com o treinamento em 100 épocas.

# Nota: A primeira época leva mais tempo do que todas as outras porque é quando o cache é criado. Depois disso, cada época leva cerca de 1 segundo

batch_size = 64
epochs = 100

print('[INFO] treinando o modelo...')
model = audio_classifier.create(
    train_data,
    spec,
    validation_data,
    batch_size=batch_size,
    epochs=epochs)

#%%
'''Avaliando o modelo nos áudios de TESTE'''
test_accuracy = model.evaluate(test_data)

# test_accuracy[0] = valor da loss
# test_accuracy[1] = valor da accuracy
print(f"[INFO] accuracy: {test_accuracy[1]*100:.2f}%")

#%%
# Visualizando a matriz de confusão utilizando o Model Marker
def show_confusion_matrix(confusion, test_labels):
  '''Calcule a matriz de confusão e normalize'''
  confusion_normalized = confusion.astype("float") / confusion.sum(axis=1)
  axis_labels = test_labels
  ax = sns.heatmap(
      confusion_normalized, xticklabels=axis_labels, yticklabels=axis_labels,
      cmap='Blues', annot=True, fmt='.2f', square=True)
  plt.title("Matriz de confusão")
  plt.ylabel("Classe real")
  plt.xlabel("Classe predita")

confusion_matrix = model.confusion_matrix(test_data)
show_confusion_matrix(confusion_matrix.numpy(), test_data.index_to_label)

#%%
'''Testando o modelo'''
# testar o modelo em um áudio de amostra do conjunto de dados de teste apenas para ver os resultados.
# Primeiro criamos o serving_model, que é o modelo de serviço.
serving_model = model.create_serving_model()

print(f'[INFO] forma e tipo de entrada do modelo: {serving_model.inputs}')
print(f'[INFO] forma e tipo de saída do modelo: {serving_model.outputs}')

# %%
def test_model():
  # Carregando um áudio aleatório
  random_audio = get_random_audio_file()
  get_bird_data(random_audio, plot = False)
  
  # Carregar o áudio com o librosa e normalizar a frequência para 16Hz
  wav_data, sample_rate = librosa.load(random_audio, sr=16000)

  # O modelo criado possui uma janela de entrada fixa.
  input_size = serving_model.input_shape[1]
  print(f'[INFO] tamanho fixo da janela: ', input_size)

  # Para um determinado arquivo de áudio, você terá que dividi-lo em partes de dados do   tamanho esperado (input_size). A última parte pode precisar ser preenchida com zeros.
  # Fazemos isto usando a função tf.signal.frame() para expandir a entrada em quadros:
  # Parâmetros:
  #   sinal: O tensor de entrada a ser expandido.
  #   frameLength: o comprimento do frame
  #   frameStep: o tamanho do salto do frame nas amostras
  #   padEnd: se deve preencher o final do sinal com padValue.
  #   padValue: Um número a ser usado onde o sinal de entrada não existe quando padEnd é  True.
  # Exemplo: 
  # o tamanho fixo da janela de entrada é 15600
  # tamanho original dos dados de um áudio é 421568
  # usando a função tf.signal.frame fazemos a divisão de 421568/15600 = 27.02 janelas/  partes, porém com o parâmetro pad_end = True completamos a última parte (faz um   arredondamento) para 28 completando a última parte com 0 (zeros). Podemos testar isto   mudando o parâmetro pad_end para False
  splitted_audio_data = tf.signal.frame(wav_data, input_size, input_size, pad_end=True,   pad_value=0)

  print(f'[INFO] caminho de áudio de teste: {random_audio}')
  print(f'[INFO] tamanho original dos dados de áudio: {len(wav_data)}')
  print(f'[INFO] número de partes para inferência: {len(splitted_audio_data)}')

  # Você fará um loop por todo o áudio dividido e aplicará o modelo para cada um deles. 

  # O modelo que você acabou de treinar tem 2 saídas: 
  # 1 - a saída do YAMNet original é mais genérica do modelo básico usado, neste caso YAMNet e a que  você acabou de treinar. Isso é importante porque o ambiente do mundo real é mais   complicado do que apenas sons de pássaros. Você pode usar a saída do YAMNet para  filtrar áudio não relevante, por exemplo, no caso de uso de pássaros, se o YAMNet não  estiver classificando pássaros ou animais, isso pode mostrar que a saída de seu modelo   pode ter uma classificação irrelevante.
  # 2 - A saída secundária que é específica para as aves que você usou no treinamento.

  # Abaixo, os dois outpus estão impressos para facilitar o entendimento de sua relação.  A maioria dos erros que seu modelo comete é quando a previsão do YAMNet não está relacionada ao seu domínio (por exemplo: pássaros).
  print(random_audio)

  results = []
  print(f'[INFO] resultado por janela de áudio: previsão com o modelo criado ->   pontuação,  (previsão YAMNET -> pontuação)')
  for i, data in enumerate(splitted_audio_data):
    yamnet_output, inference = serving_model(data) # yamnet_output: resultado do modelo   YAMNet, inference: resultado do modelo criado
    results.append(inference[0].numpy()) #o método .numpy() converte um tensor em um  objeto numpy.ndarray. Isso significa implicitamente que o tensor convertido agora  será processado na CPU.
    result_index = tf.argmax(inference[0]) # retornar a classe do modelo criado com o   valor de previsão mais alto
    spec_result_index = tf.argmax(yamnet_output[0]) # retornar a classe do modelo YAMNet  com o valor de previsão mais alto
    yamnet_labels = spec._yamnet_labels()[spec_result_index] # Recuperar as labels de   modelo YAMNet
    birds_labels = test_data.index_to_label[result_index] # Recuperar as labels em cada   indice de classe prevista com o modelo criado usnado test_data.index_to_label [result_index]

    birds_results = inference[0][result_index].numpy() # Resultado com a pontuação de   precisão da previsão
    yamnet_results = yamnet_output[0][spec_result_index]
    # Criar uma string com os resultados
    # \t para adicionar um TAB
    # A ESQUERDA: Previsões com o modelo criado e retornar a pontuação de precisão da   previsão
    # A DIREITA: Previsões com o YAMNET e pontuação de precisão da previsão
    result_str = f'[INFO] resultado da janela {i}: ' f'\t{birds_labels} -> {birds_results   * 100:.3f}%, '  f'\t({yamnet_labels} -> {yamnet_results * 100:.3f}%)'
    print(result_str)

  # Converter os resultados de previsão do modelo em uma matriz numpy, assim conseguimos  obter a classe que aparece em média mais vezes nas previsões
  results_np = np.array(results) 
  mean_results = results_np.mean(axis=0)
  result_index = mean_results.argmax()
  bird_label = test_data.index_to_label[result_index]
  bird_result = mean_results[result_index]
  print(f'[INFO] resultado médio: {bird_label} -> {bird_result * 100:.3f}%')

#%%
test_model()
# %%
# https://www.tensorflow.org/lite/tutorials/model_maker_image_classification
# Os formatos de exportação permitidos podem ser um ou uma lista dos seguintes:
# ExportFormat.TFLITE
# ExportFormat.LABEL
# ExportFormat.SAVED_MODEL

# model.export('saved_models/birds_models', export_format=[mm.ExportFormat.SAVED_MODEL, mm.ExportFormat.LABEL])
# model.save('saved_models/medical_trial_model.h5')
# %%
# Próximo passo é carregar o modelo
path_model = 'saved_models/birds_models/saved_model'
loaded = tf.saved_model.load(path_model)

#%%
loaded

# %%
print(list(loaded.signatures.keys()))
# %%
infer = loaded.signatures["serving_default"]
# print(infer.inputs)
# print(infer.outputs)

# print(f'[INFO] forma e tipo de entrada do modelo: {infer.inputs}')
print(f'[INFO] forma e tipo de saída do modelo: {infer.outputs}')

#%%
input_size = len(infer.outputs)
# input_size = serving_model.input_shape[1]
input_size

#%%
import tensorflow.keras as keras
new_model = keras.models.load_model(
    path_model, custom_objects=None, compile=False, options=None
)
new_model.summary()

# #%%
# new_model.predict(test_data)

# #%%

# #%%
# new_model.get_config()

# #%%
# new_model.optimizer

#%%
# opt = tf.keras.optimizers.RMSprop(learning_rate=0.1)
new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])


#%%
test_accuracy = new_model.evaluate(test_data)

# test_accuracy[0] = valor da loss
# test_accuracy[1] = valor da accuracy
print(f"[INFO] accuracy: {test_accuracy[1]*100:.2f}%")

# #%%
# model.name
# # %%
# loaded


# # %%
# loaded.prune()
# # %%
# labels = []

# # These are set to the default names from exported models, update as needed.
# filename = "saved_models/birds_models/saved_model/saved_model.pb"
# labels_filename = "saved_models/birds_models/labels.txt"

# # Import the TF graph
# with tf.compat.v1.Session(graph = tf.Graph()) as sess:
#    tf.compat.v1.saved_model.loader.load(
#       sess, [tf.compat.v1.saved_model.tag_constants.SERVING], "saved_models/birds_models/saved_model")

# # Create a list of labels.
# with open(labels_filename, 'rt') as lf:
#     for l in lf:
#         labels.append(l.strip())
# # %%
# model
# # %%
# with tf.compat.v1.Session() as sess:
#     input_tensor_shape = sess.graph.get_tensor_by_name('T:0').shape.as_list()
# network_input_size = input_tensor_shape[1]
# # %%

# %%
