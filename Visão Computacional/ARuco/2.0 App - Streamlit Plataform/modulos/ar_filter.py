import cv2
import numpy as np

from common import media_utils


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