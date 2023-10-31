from random import randint
from modulos import config_local as config

def generate_tone():
    color_value = [randint(0, 255)]
    color = (color_value[0], color_value[0], color_value[0])
    return color


def process_video_detection(
    cv2,
    np,
    video_index,
    demo,
    detection_demo,
    frame_count_demo,
    aug_demo,
    corners,
    ids,
    img,
):
    if not detection_demo[video_index]:
        demo[video_index].set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_count_demo[video_index] = 0
        detection_demo[video_index] = True
    else:
        if frame_count_demo[video_index] == demo[video_index].get(
            cv2.CAP_PROP_FRAME_COUNT
        ):
            demo[video_index].set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_count_demo[video_index] = 0

        _, aug_demo[video_index] = demo[video_index].read()

    img = arucoAug(
        cv2, np, bbox=corners, id=ids[0], img=img, imgAug=aug_demo[video_index]
    )

    frame_count_demo[video_index] += 1

    return img


def arucoAug(cv2, np, bbox, id, img, imgAug):
    COLOR_LINES = generate_tone()

    # Definir a margem para aumentar a região onde a imagem de sobreposição será aplicada
    # h: height - altura
    # w: width - largura
    h, w, c = img.shape
    margin_h = h*config.PROPORCAO
    margin_w = w*config.PROPORCAO

    # Definir as coordenadas dos cantos do retângulo delimitador (bbox)
    # em torno do marcador ArUco. Cada linha atribui um par de coordenadas (x, y)
    # para cada canto.

    # superior esquerdo, superior direito
    tl = bbox[0][0][0], bbox[0][0][1]
    tr = bbox[0][1][0], bbox[0][1][1]

    # inferior direito, inferior esquerdo
    br = bbox[0][2][0], bbox[0][2][1]
    bl = bbox[0][3][0], bbox[0][3][1]

    # Aumentar a região delimitada pelo marcador ArUco usando as margens calculadas
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

    # Desenhar borda sobre a imagem
    drawn_detection(cv2, np, img, pts1, COLOR_LINES, config.THICKNESS)

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
    cv2.fillConvexPoly(img, pts1.astype(int), (0, 0, 0))

    # Adicionar a imagem de sobreposição à imagem original
    imgout = cv2.addWeighted(img, 1-config.ALPHA, imgout, config.BETA, config.GAMMA)

    return imgout


def drawn_detection(cv2, np, img, pts, color_lines, thickness):
    # Parâmetros do poligono da borda
    isClosed = True

    cv2.polylines(
        img,
        [pts.astype(np.int32)],
        isClosed,
        config.COLOR_BACKGROUND_1,
        config.THICKNESS_BACKGROUND_1,
        cv2.LINE_AA,
    )
    # cv2.polylines(
    #     img,
    #     [pts.astype(np.int32)],
    #     isClosed,
    #     config.COLOR_BACKGROUND_2,
    #     config.THICKNESS_BACKGROUND_2,
    #     cv2.LINE_AA,
    # )
    # cv2.polylines(
    #     img, [pts.astype(np.int32)], isClosed, color_lines, thickness, cv2.LINE_AA
    # )

    


def drawn_line(cv2, np, l1_inicial, l1_final, text, imgout, color_lines, thickness):
    cv2.line(
        imgout,
        l1_inicial,
        l1_final,
        color_lines,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )

    # Exibir uma mensagem na ponta da linha
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, thickness)
    # text_position = (l1_final[0] - text_size[0] // 2, l1_final[1] + text_size[1])
    text_position = (l1_final[0], l1_final[1])

    cv2.putText(
        imgout,
        text,
        text_position,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1.5,
        color=color_lines,
        thickness=4,
        lineType=cv2.LINE_AA,
    )

    # Adicionar Overlay
    overlay = imgout.copy()
    cv2.rectangle(
        overlay,
        (text_position[0], text_position[1] - text_size[1]),
        (text_position[0] + text_size[0], text_position[1]),
        config.COLOR_OVERLAY,
        -1,
    )
    imgout = cv2.addWeighted(overlay, 1-config.ALPHA, imgout, config.ALPHA, 0, imgout)

    return imgout
