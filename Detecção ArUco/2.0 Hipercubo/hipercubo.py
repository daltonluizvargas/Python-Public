# %%
"""
    Antes de iniciar a detecção, 
    iniciar o módulo straemer e definir a fonte do vídeo
"""

import cv2 as cv
import time

from modulos import detectar_aruco
from modulos import apresentacao_v2

list_id_aruco = [10, 24, 2, 7]
video_path = "video/"

# Loop infinito para reiniciar o detector
while True:
    result = detectar_aruco.detector(show_detection=True, id_aruco=list_id_aruco)

    if result:
        apresentacao_v2.carregar_apresentacao(
            video_path + str(result[1]) + ".mp4", result[1], list_id_aruco
        )
