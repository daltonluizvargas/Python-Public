# %%
import time
import tensorflow as tf
import numpy as np

import librosa
import librosa.display as ld

import pyaudio, wave

import matplotlib.pyplot as plt

import netron

import seaborn as sns
sns.set()

# Função abaixo é opcional
# Salvar os registros de cada detecção em arquivo CSV para posterior análise
# Por exemplo, carro 3 foi detectado as 00:00 em 1 de abril de 2021
# https://realpython.com/python-csv/#writing-csv-files-with-csv
# https://docs.python.org/pt-br/3/library/csv.html
import csv
fp = open('results/report.csv', mode='w')        
writer_CSV = csv.DictWriter(fp, fieldnames=['time', 'result'])
writer_CSV.writeheader()

#%%
MODEL_TYPES = ['GÊNERO', 'EMOÇÃO', 'SOM']

#%%
MODEL_TYPES[2]

#%%
WAVE_OUTPUT_FILENAME = 'recordings/Audio_.wav'

#%%
def load_model_by_name(model_type, show_model = True):
    if model_type == MODEL_TYPES[0]:
        model = tf.keras.models.load_model('saved_models/genre_recognition.hdf5')
        model_dict = sorted(list(['feminino', 'masculino']))
        if show_model:
            netron.start('saved_models/genre_recognition.hdf5')
    if model_type == MODEL_TYPES[1]:
        model = tf.keras.models.load_model('saved_models/emotion_recognition.hdf5')
        model_dict = sorted(list(['neutra', 'calma', 'feliz', 'triste', 'nervosa', 'medo', 'nojo', 'surpreso']))
        if show_model:
            netron.start('saved_models/emotion_recognition.hdf5')
    if model_type == MODEL_TYPES[2]:
        model = tf.keras.models.load_model('saved_models/audio_classification.hdf5')
        model_dict = sorted(list(['ar_condicionado', 'buzina_de_carro', 'crianca_brincando', 'latido_de_cachorro', 'perfuracao', 'motor_em_marcha_lenta', 'tiro_de_arma', 'britadeira', 'sirene', 'musica_de_rua']))
        if show_model:
            netron.start('saved_models/audio_classification.hdf5')
    
    return model, model_dict
    

#%%
model_type = 'EMOÇÃO'
loaded_model = load_model_by_name(model_type, show_model=False)

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
def predictSound(AUDIO, plot = True, wave_plot = True, report = True):
    results = []
    wav_data, sample_rate = librosa.load(AUDIO) 
    
    clip, index = librosa.effects.trim(wav_data, top_db=60, frame_length=512,   hop_length=64) # O valor do parâmetro top_db foi selecionado de forma empírica para este exemplo
    
    splitted_audio_data = tf.signal.frame(clip, sample_rate, sample_rate,   pad_end=True,  pad_value=0)
    
    print(f'[INFO] tamanho original dos dados de áudio: {len(wav_data)}')
    print(f'[INFO] frequência: {(sample_rate)}')
    print(f'[INFO] número de partes para inferência: {len(splitted_audio_data)}')    

    for i, data in enumerate(splitted_audio_data.numpy()):
        mfccs_features = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)
        mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
        mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
        mfccs_scaled_features = mfccs_scaled_features[:,:,np.newaxis]

        predictions = loaded_model[0].predict(mfccs_scaled_features, batch_size=32)
        if plot: # Previsões para cada parte do áudio
            plt.figure(figsize=(len(splitted_audio_data), 5))
            plt.barh(loaded_model[1], predictions[0])  
            plt.tight_layout()
            plt.show()  

        predictions = predictions.argmax(axis=1)
        predictions = predictions.astype(int).flatten()
        predictions = loaded_model[1][predictions[0]]
        results.append(predictions)

        result_str = 'PARTE ' + str(i) + ': ' + str(predictions).upper()        
        print(result_str)

    count_results = [[results.count(x), x] for x in set(results)]
    print(count_results)
    
    if wave_plot:
        plt.title(str(max(count_results)).upper(), size=14)
        ld.waveplot(clip, sr=sample_rate, color='r')
        plt.xlabel("Tempo (segundos) ==>")
        plt.ylabel("Amplitude")    

    else:
        print(f'[INFO] classe com maior número de classificações: {max(count_results)}')   

    # if report:
    #     writer_CSV.writerow({'time': time.strftime("%c"), 'result': + str([[x,results.count(x)] for x in set(results)])}) # Gravar o resultado em  CSV

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

# %%
# recordAudio(record_seconds=5)
predictSound('testes/triste.wav', plot = False, wave_plot = False)
# %%
