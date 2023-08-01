#%%
'''
    Este scrip faz a detecção das TAGs aruco cadastradas no dicionário dict_ids,
    executa o comando para abrir a porta,
    e por fim diz um 'olá' para a pessoa com a TAG correspondente.
    
    Aguarda 6 segundos para realizar uma nova detecção.
'''
print('[INFO] Carregando bibliotecas...')

# Importar bibliotecas
import cv2 as cv
import time

# Para executar arquivos de áudio
from playsound import playsound

# Importar módulos
from modulos import detectar_aruco

# Dicionário de TAGs
dict_ids = {15: 'Carro 1'}

print('[INFO] Automação iniciada...')

# Loop infinito para reiniciar o detector
while True:
    detectar_aruco.detector(show_detection=True, id_aruco = list(dict_ids.keys())) # Chama o detector de TAGs aruco
     
    
    
        

    