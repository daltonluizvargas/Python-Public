# %%
from sklearn.exceptions import DataConversionWarning
import tensorflow as tf
import numpy as np

import pathlib
import librosa.display as ld

import pyaudio, wave

import matplotlib.pyplot as plt

import netron
import os

import seaborn as sns
sns.set()

#%%
WAVE_OUTPUT_FILENAME = 'Datasets/mini_speech_commands/up/Audio_.wav'

#%%
def load_model_by_name(show_model = False):
    model = tf.keras.models.load_model('saved_models/voice_command_recognition.hdf5')
    model_dict = ['down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes']
    if show_model:
        netron.start('saved_models/voice_command_recognition.hdf5')    
    return model, model_dict
    

#%%
loaded_model = load_model_by_name()

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

#%%
'''Pré-processamento de áudio'''
AUTOTUNE = tf.data.AUTOTUNE

def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary) 
    return tf.squeeze(audio, axis=-1)

def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2]

def get_waveform_and_label(audio):
    waveform = decode_audio(audio)
    return waveform

def get_spectrogram(waveform): 
    zero_padding = tf.zeros([44000] - tf.shape(waveform), dtype=tf.float32)
    waveform = tf.cast(waveform, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(equal_length, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    return spectrogram

def get_spectrogram_and_label_id(audio, label):
    spectrogram = get_spectrogram(audio)
    spectrogram = tf.expand_dims(spectrogram, -1)
    label_id = tf.argmax(label == loaded_model[1])
    return spectrogram, label_id

def preprocess(files):
  files_ds = tf.data.Dataset.from_tensor_slices(files)
  output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
  output_ds = output_ds.map(get_spectrogram_and_label_id,  num_parallel_calls=AUTOTUNE)

  return output_ds

# %%
import librosa
def predict(audio):
    wav_data, sample_rate = librosa.load(audio) 
    prob=loaded_model[0].predict(wav_data.reshape(1,8000,1))
    index=np.argmax(prob[0])
    return index


def recordAudio(record_seconds = 10):
    CHUNK = 8192
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
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

# %%
# recordAudio(record_seconds=2)
DATASET_MINI_SPEECH = pathlib.Path('Datasets/mini_speech_commands')
data_dir = pathlib.Path(DATASET_MINI_SPEECH)
sample_file = data_dir/'up/01b4757a_nohash_0.wav'
predict(sample_file)
# %%
