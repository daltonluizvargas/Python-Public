from modulos import config_local as config

def arucoAug(cv2, np, bbox, id, img, imgAug):
    # Definir as coordenadas dos cantos do retângulo delimitador (bbox)
    # em torno do marcador ArUco. Cada linha atribui um par de coordenadas (x, y)
    # para cada canto.

    # Definir a margem para aumentar a região onde a imagem de sobreposição será aplicada
    h, w, c = img.shape
    margin_h = h*config.PROPORCAO
    margin_w = w*config.PROPORCAO

    # superior esquerdo, superior direito
    tl = bbox[0][0][0], bbox[0][0][1]
    tr = bbox[0][1][0], bbox[0][1][1]

    # inferior direito e inferior esquerdo
    br = bbox[0][2][0], bbox[0][2][1]
    bl = bbox[0][3][0], bbox[0][3][1]

    # Aumentar a região delimitada pelo marcador ArUco usando a margem
    tl = (tl[0] - margin_h, tl[1] - margin_h)
    tr = (tr[0] + margin_h, tr[1] - margin_h)
    br = (br[0] + margin_w, br[1] + margin_w)
    bl = (bl[0] - margin_w, bl[1] + margin_w)

    # Extrair as dimensões da imagem de sobreposição (imgAug) em termos de altura (h),
    # largura (w) e número de canais (c).
    h, w, c = imgAug.shape

    # Criar dois conjuntos de pontos:
    # pts1 representa os cantos do retângulo do marcador ArUco no mundo real
    # pts2 representa os cantos correspondentes no plano da imagem de sobreposição.
    pts1 = np.array([tl, tr, br, bl])
    pts2 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    # Calcular a matriz de homografia que mapeia os pontos de pts2 para pts1.
    # A matriz de homografia permite mapear os pontos da imagem de sobreposição no espaço do marcador ArUco.
    matrix, _ = cv2.findHomography(pts2, pts1)

    # Aplicar a transformação de perspectiva na imagem de sobreposição (imgAug)
    # usando a matriz de homografia (matrix).
    # A imagem resultante é armazenada em imgout.
    imgout = cv2.warpPerspective(imgAug, matrix, (img.shape[1], img.shape[0]))

    # Preencher o polígono definido pelos pontos pts1 no retângulo delimitador do marcador
    # ArUco com a cor preta (0, 0, 0) na imagem original (img).
    # Isso é feito para remover o conteúdo da região onde o marcador será sobreposto.
    # cv2.fillConvexPoly(img, pts1.astype(int), (0, 0, 0))

    # Adicionar a imagem de sobreposição à imagem original
    imgout = cv2.addWeighted(img, 1-config.ALPHA, imgout, config.ALPHA, 0)
    # imgout = img + imgout

    return imgout
