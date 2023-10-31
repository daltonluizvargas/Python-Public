"""
https://colab.research.google.com/drive/1iKoX8Qn3KBu3DChN8GqCE6mh5ZSP2DW8?usp=sharing#scrollTo=HkWBmT9DjLDo
"""

# Importando as bibliotecas necessárias
import cv2
import numpy as np
from common import media_utils

# Definindo o nome da imagem que deseja usar como imagem de origem
src_image_index = 'zequinha.png'

# Definindo as dimensões desejadas para o vídeo capturado
width, height = 800, 600

# Inicializando a captura de vídeo da câmera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# Configurando as dimensões do vídeo capturado
cap.set(3, width)
cap.set(4, height)

# Definindo a função para configurar os elementos da imagem de origem
def set_src_image(image_path):
    global src_image, src_image_gray, src_mask, src_landmark_points, src_np_points, src_convexHull, indexes_triangles
    # Carregando a imagem de origem
    src_image = cv2.imread(image_path)
    # Convertendo a imagem para escala de cinza
    src_image_gray = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
    # Criando uma máscara preta para a imagem de origem
    src_mask = np.zeros_like(src_image_gray)
    
    # Obtendo os pontos de referência faciais da imagem de origem
    src_landmark_points = media_utils.get_landmark_points(src_image)
    # Convertendo os pontos de referência para um array numpy
    src_np_points = np.array(src_landmark_points)
    # Calculando a convexHull dos pontos de referência
    src_convexHull = cv2.convexHull(src_np_points)
    # Preenchendo a convexHull com branco na máscara
    cv2.fillConvexPoly(src_mask, src_convexHull, 255)
    
    # Obtendo os triângulos dos pontos de referência
    indexes_triangles = media_utils.get_triangles(convexhull=src_convexHull,
                                                  landmarks_points=src_landmark_points,
                                                  np_points=src_np_points)
    
# Configurando a imagem de origem
set_src_image(f"images/{src_image_index}")

while True:
    # Atualizando as variáveis globais da imagem de origem
    global src_image, src_image_gray, src_mask, src_landmark_points, src_np_points, src_convexHull, indexes_triangles

    # Capturando um frame da câmera
    _, dest_image = cap.read()
    # Redimensionando o frame capturado para as dimensões desejadas
    dest_image = cv2.resize(dest_image, (width, height))

    # Convertendo a imagem de destino para escala de cinza
    dest_image_gray = cv2.cvtColor(dest_image, cv2.COLOR_BGR2GRAY)
    # Criando uma máscara preta para a imagem de destino em escala de cinza
    dest_mask = np.zeros_like(dest_image_gray)

    # Obtendo os pontos de referência faciais da imagem de destino
    dest_landmark_points = media_utils.get_landmark_points(dest_image)
    # Se não forem encontrados pontos de referência na imagem de destino, continue para o próximo loop
    if dest_landmark_points is None:
        continue
    dest_np_points = np.array(dest_landmark_points)
    # Calculando a convexHull dos pontos de referência da imagem de destino
    dest_convexHull = cv2.convexHull(dest_np_points)

    # Obtendo as dimensões da imagem de destino
    height, width, channels = dest_image.shape
    # Criando uma imagem vazia para a nova face
    new_face = np.zeros((height, width, channels), np.uint8)

    # Realizando a triangulação de ambos os rostos
    for triangle_index in indexes_triangles:
        # Triangulação do primeiro rosto (imagem de origem)
        points, src_cropped_triangle, cropped_triangle_mask, _ = media_utils.triangulation(
            triangle_index=triangle_index,
            landmark_points=src_landmark_points,
            img=src_image)

        # Triangulação do segundo rosto (imagem de destino)
        points2, _, dest_cropped_triangle_mask, rect = media_utils.triangulation(triangle_index=triangle_index,
                                                                                 landmark_points=dest_landmark_points)

        # Aplicando a transformação de warp aos triângulos
        warped_triangle = media_utils.warp_triangle(rect=rect, points1=points, points2=points2,
                                                    src_cropped_triangle=src_cropped_triangle,
                                                    dest_cropped_triangle_mask=dest_cropped_triangle_mask)

        # Reconstruindo o rosto de destino
        media_utils.add_piece_of_new_face(new_face=new_face, rect=rect, warped_triangle=warped_triangle)

    # Realizando a substituição de rostos (face swapping)
    result = media_utils.swap_new_face(dest_image=dest_image, dest_image_gray=dest_image_gray,
                                       dest_convexHull=dest_convexHull, new_face=new_face)

    # Aplicando um filtro de mediana para suavizar a imagem resultante
    result = cv2.medianBlur(result, 3)
    
    # Redimensionando a imagem de origem para exibição
    h, w, _ = src_image.shape
    rate = width / w

    # Exibindo as imagens na janela
    # cv2.imshow("Source image", cv2.resize(src_image, (int(w * rate), int(h * rate))))
    cv2.imshow("New face", new_face)
    cv2.imshow("Result", result)

    # Capturando entrada do teclado
    key = cv2.waitKey(3)
    # Encerrando o loop se a tecla ESC for pressionada
    if key == 27:
        break

# Liberando recursos e fechando as janelas
cap.release()
cv2.destroyAllWindows()

