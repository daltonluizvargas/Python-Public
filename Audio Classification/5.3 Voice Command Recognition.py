#%%
# Built-in Python Modules 
# These are mainly for file operations
import glob
import io
import os
import pathlib
import random
import pyaudio, wave

import seaborn as sns
sns.set()

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

#%%
# Dataset directories
speech_data = pathlib.Path('Datasets/mini_speech_commands')
audio_dir = pathlib.Path(speech_data)

#%%
speech_commands = []

# Iterate through extracted files to get the spoken commands
for name in glob.glob(str(speech_data) + '/*' + os.path.sep):
  speech_commands.append(name.split('\\')[-2])

# # Dictionary comprehension to map commands to an integer for quick lookups
speech_commands_dict = {i : speech_commands.index(i) for i in speech_commands}

speech_commands_dict

#%%
speech_data_list = []

# Iterate through spoken commands to get individual audio files
for name in glob.glob(str(speech_data) + '/*/*'):
  speech_data_list.append(name)


# # Seed to ensure shuffled data is repeatable on the same hardware
random.seed(42)
random.shuffle(speech_data_list)

#%%
# Labels for corresponding shuffled audio data
speech_data_labels = []

for audio in speech_data_list:
  speech_data_labels.append(os.path.dirname(audio).split('\\')[-1])
speech_data_labels


#%%
# Integer representation of labels based on 'speech_commands_dict'
speech_label_int = []

for audio in speech_data_labels:
  speech_label_int.append(speech_commands_dict[audio])
speech_label_int

#%%
# Compiling all speech data into a list
loaded_speech_data = []

for audio in speech_data_list:
  loaded_speech_data.append(librosa.load(audio, sr=16000))

loaded_speech_data

#%%
speech_data_mfcc = []

for loaded_audio in loaded_speech_data:
  speech_data_mfcc.append(librosa.feature.mfcc(loaded_audio[0], loaded_audio[1]))
speech_data_mfcc

#%%
example_index = 5
librosa.display.specshow(speech_data_mfcc[example_index], x_axis='time', y_axis='hz')
plt.colorbar()
plt.tight_layout()
plt.title(f'mfcc for \"{speech_data_labels[example_index]}\"')
plt.show

#%%
waveform_example = librosa.feature.inverse.mfcc_to_audio(
    speech_data_mfcc[example_index])
librosa.display.waveplot(waveform_example)
plt.tight_layout()
plt.title(f'waveform for \"{speech_data_labels[example_index]}\"')
plt.show

#%%
data_length = len(speech_data_list)
data_ratio = {
    'train': 0.7,
    'validate': 0.15,
    'test': 0.15
}
training_ratio = int(data_length*data_ratio['train'])
validation_ratio = int(data_length*data_ratio['validate'])
testing_ratio = int(data_length*data_ratio['test'])

print(f"Dataset Ratio - Training Data: {data_length*data_ratio['train']:.0f}, \
Validation Data: {data_length*data_ratio['validate']:.0f}, Testing Data: \
{data_length*data_ratio['test']:.0f}")

#%%
speech_data_as_tensor = []

for index in range(len(speech_data_mfcc)):
  # Inconsistency in array size is rectified by resize the array and
  # filling with zeros
  mfcc_array = np.copy(speech_data_mfcc[index])
  mfcc_array.resize((20,32), refcheck=False)
  speech_data_as_tensor.append(tf.expand_dims(
      tf.convert_to_tensor(mfcc_array), -1))

speech_data_as_tensor[0].shape

#%%
# Dataset slicing to desired ratios
training_slice = speech_data_as_tensor[:5600]
validation_slice = speech_data_as_tensor[5600: 5600 + 1200]
testing_slice = speech_data_as_tensor[5600 + 1200:]

#%%
training_dataset = tf.data.Dataset.from_tensor_slices((
    training_slice, speech_label_int[:5600]))
validation_dataset = tf.data.Dataset.from_tensor_slices((
    validation_slice, speech_label_int[5600: 5600+1200]))
testing_dataset = tf.data.Dataset.from_tensor_slices((
    testing_slice, speech_label_int[-1200:]))

batch_size = 10

# Adds a dimension to the dataset that is necessary for 
# model fit tensorflow function
training_dataset = training_dataset.batch(batch_size)
validation_dataset = validation_dataset.batch(batch_size)

#%%
num_labels = len(speech_commands)
norm_layer = layers.Normalization()

# CNN model code as from source [1]
model = models.Sequential([
    layers.Input(shape=(20,32,1)),
    layers.Resizing(32, 32), 
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
# CNN model compile code as from source [1]
model.compile(
    optimizer='Adam',
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
  tf.keras.callbacks.ModelCheckpoint(filepath='saved_models/voice_command_recognition_testando_novo.hdf5', save_best_only=True)
)
history = model.fit(
    training_dataset, 
    validation_data=validation_dataset,  
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
test_audio_data = []
test_label_data = []

for audio, label in testing_dataset:
  test_audio_data.append(audio.numpy())
  test_label_data.append(label.numpy())

test_audio_data = np.array(test_audio_data)
test_label_data = np.array(test_label_data)

predicted_values = np.argmax(model.predict(test_audio_data), axis=1)
true_values = test_label_data

test_accuracy = sum(predicted_values == true_values) / len(true_values)
print(f'[INFO] precisão no conjunto de teste: {test_accuracy:.0%}')

#%%
'''Exibir uma matriz de confusão'''
# Use uma matriz de confusão para verificar quão bem o modelo classifica cada um dos comandos do conjunto de teste
confusion_mtx = tf.math.confusion_matrix(true_values, predicted_values) 
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx, xticklabels=speech_commands_dict, yticklabels=speech_commands_dict, 
            annot=True, fmt='g', cmap='coolwarm')
plt.xlabel('Classes Previstas')
plt.ylabel('Calsses Reais')
plt.show()


#%%
'''Executar inferência em um arquivo de áudio'''
# Finalmente, verifique a saída de previsão do modelo usando um arquivo de áudio de entrada de alguém dizendo "não"
def predict(sample_file, plot = False):
  audio, sample_rate = librosa.load(sample_file, sr = 16000)
  mfcc = librosa.feature.mfcc(audio)
  
  # A inconsistência no tamanho da matriz é corrigida redimensionando amatriz e   preenchendo com zeros 
  mfcc_array = np.copy(mfcc)
  mfcc_array.resize((20,32), refcheck=False)
  speech_tensor = tf.expand_dims(tf.convert_to_tensor(mfcc_array), -1)
  speech_tensor = tf.expand_dims(speech_tensor, axis=0)
  
  prediction = model(speech_tensor)
  index = np.argmax(prediction[0])
  result = f' Comando previsto "{speech_commands[index]}"'
  if plot:
    plt.bar(speech_commands, tf.nn.softmax(prediction[0]), color=list('rgbkymc')  )
    plt.title(str(result).upper())
    plt.show()
  
  else:
    print('[INFO]' + result)

WAVE_OUTPUT_FILENAME = 'recordings/Audio_.wav'
def recordAudio(record_seconds = 10):
    CHUNK = 8192
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = record_seconds
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
             channels = CHANNELS,
             rate = RATE,
             input = True,
             input_device_index = 0,
             frames_per_buffer = CHUNK)
    print("[INFO]* gravando")
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("[INFO]* gravação concluída")
    stream.stop_stream()    # "Stop Audio Recording
    stream.close()          # "Close Audio Recording
    p.terminate()           # "Audio System Close
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

print('[INFO] funções criadas')

#%%
filenames = tf.io.gfile.glob(str(speech_data) + '/*/*')
n_files = len(filenames)

rnd = np.random.randint(0, n_files)
filenames[rnd]
fname = filenames[rnd] 
command = filenames[rnd]
print(f'[INFO] arquivo de audio No. {rnd}')
print(f'[INFO] path: {fname}')
predict(fname, plot = True)

#%%
'''Carregar o modelo criado e testar em audio gravado'''
model = tf.keras.models.load_model('saved_models/voice_command_recognition_testando_novo.hdf5')

recordAudio(record_seconds=2)
predict(WAVE_OUTPUT_FILENAME)


