# %%
import tensorflow as tf
model2 = tf.keras.models.load_model('saved_models\genre_recognition.hdf5')
# %%
'''Testando em um audio'''
import numpy as np
import librosa
import librosa.display as ld
AUDIO="testes/triste.wav"
audio, sample_rate = librosa.load(AUDIO, res_type='kaiser_fast') 
mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)

predict_x=model2.predict(mfccs_scaled_features)
classes_x=np.argmax(predict_x,axis=1)

classes_x
# %%
