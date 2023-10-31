def create_cube(np):
    # Definição dos vértices do cubo em coordenadas 3D
    vertices = np.array(
        [
            [1, 1, -1],
            [1, -1, -1],
            [-1, -1, -1],
            [-1, 1, -1],
            [1, 1, 1],
            [1, -1, 1],
            [-1, -1, 1],
            [-1, 1, 1],
        ],
        dtype=np.float32,
    )

    # Definição das faces (planos) do cubo usando os vértices definidos anteriormente
    # Cada face é representada por quatro vértices, definindo um quadrado.
    # O cubo possui seis faces (uma para cada lado).
    faces = np.array(
        [
            [
                vertices[0],
                vertices[1],
                vertices[2],
                vertices[3],
            ],  # Face frontal do cubo
            [
                vertices[4],
                vertices[5],
                vertices[6],
                vertices[7],
            ],  # Face traseira do cubo
            [
                vertices[0],
                vertices[1],
                vertices[5],
                vertices[4],
            ],  # Face lateral direita
            [
                vertices[2],
                vertices[3],
                vertices[7],
                vertices[6],
            ],  # Face lateral esquerda
            [
                vertices[0],
                vertices[3],
                vertices[7],
                vertices[4],
            ],  # Face superior do cubo
            [
                vertices[1],
                vertices[2],
                vertices[6],
                vertices[5],
            ],  # Face inferior do cubo
        ],
        dtype=np.float32,
    )

    # Retorna os vértices e faces do cubo
    return vertices, faces


def render_cube(cv2, np, img, vertices, faces, projection):
    scale_matrix = np.eye(3) * 100  # Fator de escala para o tamanho do cubo
    h, w = 300, 300

    # Cores para cada face do cubo (vermelho, verde, azul, amarelo, magenta, ciano)
    colors = [
        (0, 0, 255),
        (0, 255, 0),
        (255, 0, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 0),
    ]

    for i, face in enumerate(faces):
        # Aplica a escala ao cubo, transformando os vértices
        points = np.dot(face, scale_matrix)

        # Inverte o eixo y dos vértices, para exibir o cubo de cima
        points = np.dot(points, np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]))

        # Desloca os vértices para o centro da imagem
        points = np.array([[p[0] + 150, p[1] + 150, p[2]] for p in points])

        # Aplica a perspectiva do cubo para a projeção da imagem
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)

        # Desenha as arestas do cubo com cores diferentes para cada face
        cv2.polylines(img, [imgpts], isClosed=True, color=colors[i], thickness=2)

        # Desenha os lados preenchidos do cubo com uma cor mais suave
        side_color = tuple(
            c // 20 for c in colors[i]
        )  # Cor para os lados do cubo (metade da intensidade das cores das arestas)
        cv2.fillConvexPoly(img, imgpts, side_color)

    # Define o interior do cubo como transparente (canal alfa)
    alpha_mask = np.zeros((h, w, 1), dtype=np.uint8)
    cv2.fillConvexPoly(alpha_mask, imgpts, 255)
    img[:, :, 3] = alpha_mask[:, :, 0]

    return img


def render_cube_v2(cv2, np, img, vertices, faces, projection):
    scale_matrix = np.eye(3) * 200  # Fator de escala para o tamanho do cubo
    h, w = 300, 300

    # Cores para as arestas do cubo (azul escuro)
    edge_color = (50, 50, 200)

    for i, face in enumerate(faces):
        # Aplica a escala ao cubo, transformando os vértices
        points = np.dot(face, scale_matrix)

        # Inverte o eixo y dos vértices, para exibir o cubo de cima
        points = np.dot(points, np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]))

        # Desloca os vértices para o centro da imagem
        points = np.array([[p[0] + 150, p[1] + 150, p[2]] for p in points])

        # Aplica a perspectiva do cubo para a projeção da imagem
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)

        # Desenha apenas as arestas do cubo com a cor azul escuro e espessura 3
        cv2.polylines(img, [imgpts], isClosed=True, color=edge_color, thickness=3)

    # Define o interior do cubo como transparente (canal alfa)
    alpha_mask = np.zeros((h, w, 1), dtype=np.uint8)
    cv2.fillConvexPoly(alpha_mask, imgpts, 255)
    img[:, :, 3] = alpha_mask[:, :, 0]

    return img


def projection_matrix(cv2, np, math, cam_parameters, homography):
    homography = homography * (-1)  # Inverte o sinal da homografia

    # Calcula a rotação e translação da matriz de projeção
    rot_and_trans = np.dot(np.linalg.inv(cam_parameters), homography)

    # Separa as colunas da matriz de rotação e translação
    col_1 = rot_and_trans[:, 0]
    col_2 = rot_and_trans[:, 1]
    col_3 = rot_and_trans[:, 2]

    # Calcula o fator de escala para a rotação
    ii = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))

    # Normaliza as colunas de rotação pela escala
    rot_1 = col_1 / ii
    rot_2 = col_2 / ii

    # Calcula a matriz de translação
    translation = col_3 / ii

    # Calcula os vetores c, p e d para construir a matriz de rotação final
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)

    # Constrói a matriz de rotação final
    rot_1 = np.dot(
        c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2)
    )
    rot_2 = np.dot(
        c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2)
    )

    # Calcula o vetor de rotação final
    rot_3 = np.cross(rot_1, rot_2)

    # Empilha as colunas de rotação e translação para formar a matriz de projeção final
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T

    # Multiplica a matriz de câmera pelas projeções para obter a matriz de projeção completa
    return np.dot(cam_parameters, projection)
