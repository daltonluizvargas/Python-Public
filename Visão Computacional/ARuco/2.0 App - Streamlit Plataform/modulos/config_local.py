from random import randint

TYPE = ("VIDEO", "PORTAL", "Eu, Zequinha")


'''-------------Parâmetros dos módulos de detecção ARUCO-------------'''
COLOR_OVERLAY = (245, 43, 115)
COLOR_TEXT = (245, 43, 115)
COLOR_LINES = (randint(0, 255), randint(0, 255), randint(0, 255))
TEXT_SIZE = 0.75
WEBCAM_ID = 0
THICKNESS = randint(1, 2)
PROPORCAO = .5 # porcentagem da proporção desejada sobre as dimensõe da câmera do dispositivo
ALPHA = .3
BETA = .95
GAMMA = .15
# Cor da borda das detecções
COLOR_BACKGROUND_1 = (15, 15, 15)
THICKNESS_BACKGROUND_1 = 3
COLOR_BACKGROUND_2 = (255, 0, 255)
THICKNESS_BACKGROUND_2 = 6

# Constantes para os caminhos dos vídeos
PATH_VIDEOS = "video/"
TAGS_ID = [101,103,104,105,106,107,108]


'''-------------Parâmetros do módulo PORTAL-------------'''
PATH_IMAGE_PORTAL = "image/img_f4.png"



'''-------------Parâmetros do módulo DEEPFAKE-------------'''
PATH_IMAGE_FACE = 'image/image_faces/zequinha.png'


#%%
import cv2
print(cv2. __version__)