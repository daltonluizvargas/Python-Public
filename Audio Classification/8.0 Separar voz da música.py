import numpy as np
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import librosa

sr = 44100

y, sr = librosa.load(librosa.ex('fishin'), sr = sr)
output_file_path = "testOutput.wav"

S_full, phase = librosa.magphase(librosa.stft(y))
S_filter = librosa.decompose.nn_filter(S_full,
                                       aggregate=np.median,
                                       metric='cosine',
                                       width=int(librosa.time_to_frames(2, sr=sr)))
S_filter = np.minimum(S_full, S_filter)
margin_i, margin_v = 2, 10
power = 2

mask_i = librosa.util.softmask(S_filter,
                               margin_i * (S_full - S_filter),
                               power=power)

mask_v = librosa.util.softmask(S_full - S_filter,
                               margin_v * S_filter,
                               power=power)

# Once we have the masks, simply multiply them with the input spectrum
# to separate the components
S_foreground = mask_v * S_full
S_background = mask_i * S_full
D_foreground = S_foreground * phase
y_foreground = librosa.istft(D_foreground)
sf.write(output_file_path, y_foreground, samplerate=44100, subtype='PCM_24')