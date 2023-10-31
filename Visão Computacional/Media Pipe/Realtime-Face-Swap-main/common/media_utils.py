import sys

import cv2
import mediapipe as mp
import numpy as np

# Importando o módulo de utilidades de desenho do MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# Definindo a função que obtém os pontos de referência do rosto
def get_landmark_points(src_image):
    # Inicializando o modelo FaceMesh com configurações específicas
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:
        # Convertendo a imagem BGR para RGB antes do processamento
        results = face_mesh.process(cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB))

        # Verificando e desenhando os pontos de referência do rosto na imagem
        if not results.multi_face_landmarks:
            return None
        if len(results.multi_face_landmarks) > 1:
            sys.exit("Há muitos pontos de referência do rosto")

        # Extraindo os pontos de referência do rosto detectado
        src_face_landmark = results.multi_face_landmarks[0].landmark
        landmark_points = []
        for i in range(468):
            # Calculando as coordenadas (x, y) dos pontos de referência
            y = int(src_face_landmark[i].y * src_image.shape[0])
            x = int(src_face_landmark[i].x * src_image.shape[1])
            landmark_points.append((x, y))

        return landmark_points


# Essa função, extract_index_nparray, tem como objetivo extrair o primeiro elemento de um array numpy e retorná-lo. 
# Ela percorre os elementos do primeiro elemento do array numpy, 
# atribuindo o valor do primeiro elemento encontrado à variável 'index' e, 
# em seguida, para o loop. Finalmente, ela retorna o valor do 'index', que é o primeiro elemento do array.
def extract_index_nparray(nparray):
    # Inicializando a variável 'index' como nula
    index = None
    
    # Iterando pelos elementos do primeiro elemento do array numpy (nparray[0])
    for num in nparray[0]:
        # Atribuindo o valor do elemento atual à variável 'index'
        index = num
        
        # Parando o loop após a primeira iteração para extrair apenas o primeiro elemento
        break
    
    # Retornando o valor do 'index'
    return index

# Essa função get_triangles tem como objetivo criar uma lista de triângulos definidos pelos índices dos pontos de referência faciais. 
# Ela começa encontrando o retângulo delimitador da convexhull dos pontos de referência, 
# cria uma subdivisão usando essa convexhull e insere os pontos de referência nessa subdivisão. 
# Em seguida, ela obtém a lista de triângulos resultante da subdivisão e 
# extrai os índices correspondentes dos pontos de referência faciais. 
# e os índices forem encontrados para todos os vértices do triângulo, 
# a função cria uma lista de índices para o triângulo e a adiciona à lista de triângulos a ser retornada.
def get_triangles(convexhull, landmarks_points, np_points):
    # Encontrando o retângulo delimitador da convexhull
    rect = cv2.boundingRect(convexhull)
    
    # Criando um objeto Subdiv2D a partir do retângulo
    subdiv = cv2.Subdiv2D(rect)
    
    # Inserindo os pontos de referência dos marcos faciais na subdivisão
    subdiv.insert(landmarks_points)
    
    # Obtendo a lista de triângulos da subdivisão
    triangles = subdiv.getTriangleList()
    
    # Convertendo a lista de triângulos para um array numpy com tipo de dados int32
    triangles = np.array(triangles, dtype=np.int32)

    # Inicializando uma lista para armazenar os índices dos triângulos
    indexes_triangles = []
    
    # Iterando pelos triângulos
    for t in triangles:
        # Extraindo os pontos dos vértices do triângulo
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        # Encontrando os índices dos pontos de referência correspondentes aos vértices do triângulo
        index_pt1 = np.where((np_points == pt1).all(axis=1))
        index_pt1 = extract_index_nparray(index_pt1)
        index_pt2 = np.where((np_points == pt2).all(axis=1))
        index_pt2 = extract_index_nparray(index_pt2)
        index_pt3 = np.where((np_points == pt3).all(axis=1))
        index_pt3 = extract_index_nparray(index_pt3)

        # Verificando se os índices foram encontrados para todos os vértices do triângulo
        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            # Criando uma lista de índices para o triângulo
            triangle = [index_pt1, index_pt2, index_pt3]
            # Adicionando a lista de índices à lista de triângulos
            indexes_triangles.append(triangle)

    # Retornando a lista de índices dos triângulos
    return indexes_triangles

# Essa função, triangulation, tem como objetivo realizar a triangulação de um triângulo formado pelos pontos de referência faciais. 
# Ela recebe o índice dos pontos de referência do triângulo, os pontos de referência faciais e opcionalmente uma imagem. 
# A função calcula os pontos relativos ao retângulo delimitador do triângulo, 
# recorta a parte correspondente do triângulo da imagem (se fornecida), 
# cria uma máscara preta para o triângulo recortado e retorna os pontos do triângulo, 
# a parte recortada do triângulo, a máscara e o retângulo delimitador.
def triangulation(triangle_index, landmark_points, img=None):
    # Obtendo os pontos de referência do triângulo a partir dos índices
    tr1_pt1 = landmark_points[triangle_index[0]]
    tr1_pt2 = landmark_points[triangle_index[1]]
    tr1_pt3 = landmark_points[triangle_index[2]]
    
    # Criando um array numpy com os pontos do triângulo
    triangle = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)

    # Encontrando o retângulo delimitador do triângulo
    rect = cv2.boundingRect(triangle)
    (x, y, w, h) = rect

    # Inicializando a variável para armazenar a parte recortada do triângulo na imagem
    cropped_triangle = None
    
    # Recortando a parte do triângulo da imagem se a imagem for fornecida
    if img is not None:
        cropped_triangle = img[y: y + h, x: x + w]

    # Criando uma máscara preta para o triângulo recortado
    cropped_triangle_mask = np.zeros((h, w), np.uint8)

    # Ajustando os pontos do triângulo relativos ao retângulo
    points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                       [tr1_pt2[0] - x, tr1_pt2[1] - y],
                       [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)

    # Preenchendo a máscara com os pontos do triângulo
    cv2.fillConvexPoly(cropped_triangle_mask, points, 255)

    # Retornando os pontos do triângulo, a parte recortada do triângulo, a máscara e o retângulo
    return points, cropped_triangle, cropped_triangle_mask, rect


# Essa função warp_triangle realiza a distorção de um triângulo de origem para um triângulo de destino usando uma transformação afim. 
# Ela recebe o retângulo delimitador do triângulo, os pontos de referência do triângulo de origem (points1), 
# os pontos de referência do triângulo de destino (points2), 
# a parte recortada do triângulo de origem (src_cropped_triangle) e a máscara do triângulo de destino (dest_cropped_triangle_mask). 
# A função calcula uma matriz de transformação afim usando os pontos de referência correspondentes, 
# aplica essa transformação ao triângulo de origem, utiliza a máscara para manter apenas a área 
# desejada do triângulo distorcido e retorna o triângulo distorcido resultante.
def warp_triangle(rect, points1, points2, src_cropped_triangle, dest_cropped_triangle_mask):
    # Obtendo as coordenadas do retângulo delimitador
    (x, y, w, h) = rect
    
    # Calculando a matriz de transformação afim usando os pontos correspondentes
    matrix = cv2.getAffineTransform(np.float32(points1), np.float32(points2))
    
    # Aplicando a transformação afim para distorcer o triângulo de origem
    warped_triangle = cv2.warpAffine(src_cropped_triangle, matrix, (w, h))
    
    # Aplicando a máscara do triângulo de destino para manter apenas a área do triângulo distorcido
    warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=dest_cropped_triangle_mask)
    
    # Retornando o triângulo distorcido
    return warped_triangle


# Essa função add_piece_of_new_face tem como objetivo adicionar uma parte distorcida do rosto de origem ao novo rosto. 
# Ela recebe o novo rosto (new_face), o retângulo delimitador da parte distorcida (rect) e o triângulo distorcido (warped_triangle). 
# A função recorta a área correspondente do novo rosto, converte essa área para escala de cinza, 
# cria uma máscara dos triângulos projetados usando um limiar, aplica essa máscara ao triângulo distorcido e, 
# finalmente, adiciona o triângulo distorcido à área recortada do novo rosto. 
# A área modificada é então atualizada na região correspondente do novo rosto.
def add_piece_of_new_face(new_face, rect, warped_triangle):
    # Obtendo as coordenadas do retângulo delimitador
    (x, y, w, h) = rect
    
    # Recortando a área correspondente do novo rosto
    new_face_rect_area = new_face[y: y + h, x: x + w]
    
    # Convertendo a área recortada para escala de cinza
    new_face_rect_area_gray = cv2.cvtColor(new_face_rect_area, cv2.COLOR_BGR2GRAY)
    
    # Aplicando um limiar para obter a máscara dos triângulos projetados
    _, mask_triangles_designed = cv2.threshold(new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
    
    # Aplicando a máscara dos triângulos projetados ao triângulo distorcido
    warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

    # Adicionando o triângulo distorcido à área recortada do novo rosto
    new_face_rect_area = cv2.add(new_face_rect_area, warped_triangle)
    
    # Atualizando a área do novo rosto com a área modificada
    new_face[y: y + h, x: x + w] = new_face_rect_area


# Essa função swap_new_face realiza a substituição de uma parte do rosto de destino pelo novo rosto. 
# Ela recebe a imagem de destino (dest_image), a versão em escala de cinza da imagem de destino (dest_image_gray), 
# a convexHull do rosto de destino (dest_convexHull) e a nova parte do rosto (new_face). 
# A função cria máscaras para a parte do rosto de destino e para a parte a ser substituída, 
# remove a parte a ser substituída da imagem de destino, adiciona a nova parte do rosto e 
# realiza uma clonagem sem costura da parte substituída para a imagem de destino. 
# O centro do rosto de destino é usado como ponto de referência para a clonagem.
def swap_new_face(dest_image, dest_image_gray, dest_convexHull, new_face):
    # Criando uma máscara preta para o rosto de destino
    face_mask = np.zeros_like(dest_image_gray)
    
    # Preenchendo a máscara com a convexHull do rosto de destino
    head_mask = cv2.fillConvexPoly(face_mask, dest_convexHull, 255)
    
    # Criando uma máscara invertida para a parte do rosto que será substituída
    face_mask = cv2.bitwise_not(head_mask)

    # Criando a imagem do rosto de destino sem a parte a ser substituída
    head_without_face = cv2.bitwise_and(dest_image, dest_image, mask=face_mask)
    
    # Adicionando a nova parte do rosto à imagem sem a parte substituída
    result = cv2.add(head_without_face, new_face)

    # Encontrando o centro do rosto de destino para clonagem
    (x, y, w, h) = cv2.boundingRect(dest_convexHull)
    center_face = (int((x + x + w) / 2), int((y + y + h) / 2))

    # Realizando a clonagem sem costura da parte substituída para a imagem de destino
    return cv2.seamlessClone(result, dest_image, head_mask, center_face, cv2.NORMAL_CLONE)

