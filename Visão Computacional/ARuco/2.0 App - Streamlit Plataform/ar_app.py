# Importação das bibliotecas
import av
import cv2
import numpy as np
from random import randint

import streamlit as st
from streamlit_webrtc import (
    VideoProcessorBase,
    webrtc_streamer,
    RTCConfiguration,
)

# Importação dos módulos
from modulos import (
    ar_video, 
    ar_image,
    config_local as config
)
from modulos.common import media_utils

st.set_page_config(
    page_title="AR Journey",
    page_icon="🧊",
)

# st.balloons()

# Constantes para os caminhos dos vídeos
TYPE = st.selectbox("Escolha o modo de AR: ", config.TYPE)
    

if TYPE == 'Eu, Zequinha': 
    with st.expander('Instruções de uso'):
        st.caption('**Passo 1:** Toque no botão :blue[**START**] para iniciar.')
        st.caption('**Passo 2:** Aponte a câmera para uma face.')
else:
    with st.expander('Instruções de uso'):
        st.markdown('**Passo 1:** Selecione a câmera **TRASEIRA** clicando no botão **SELECT DEVICE** e depois na câmera com a descrição *facing back*.')
        st.divider()
        st.markdown('**Passo 2:** Toque no botão :blue[**START**] para iniciar.')
        st.divider()
        st.markdown('**Passo 3:** Aponte para uma **TAG ARuco**.')

VIDEOS = [f"{config.PATH_VIDEOS}{id}.mp4" for id in config.TAGS_ID]
qtd_videos = len(VIDEOS)


# %%
class VideoTransformer(VideoProcessorBase):
    def __init__(self):
        # Definir a imagem do filtro personalizado
        self.src_image_index = config.PATH_IMAGE_FACE
        self.setup_src_image() # Chamar o método para configurar a imagem de origem


        # Variáveis de controle e inicialização dos vídeos
        # Lista que contém a quantidade de elementos (baseada na quantidade de vídeos carregados) inicializados como False
        self.detection_demo = [False] * qtd_videos
        # Lista que contém a quantidade de elementos (baseada na quantidade de vídeos carregados)
        self.frame_count_demo = [0] * qtd_videos
        # Compreensão de lista, uma forma concisa de criar uma lista aplicando uma expressão a cada item em outra lista (ou iterável).
        # Neste caso, está sendo aplicada a função cv2.VideoCapture(video) a cada item video na lista VIDEOS
        self.demo = [cv2.VideoCapture(video) for video in VIDEOS]

        # Ler imagem do portal
        self.imagAugPortal = cv2.imread(config.PATH_IMAGE_PORTAL)

        self.translation_factor = 0.0

        # Verificar se os vídeos foram abertos corretamente
        if not all(video.isOpened() for video in self.demo):
            for video in self.demo:
                video.release()
            st.warning("O vídeo de AR não pode ser carregado!", icon="⚠️")

        # Ler o primeiro frame de cada vídeo
        # Inicilizar a lista de vídeos aug como None, indicando que o primeiro frame ainda não foi carregado
        self.aug_demo = [None] * qtd_videos
        # Ler o primeiro frame de cada vídeo carregado
        for i, video in enumerate(self.demo):
            _, self.aug_demo[i] = video.read()

    # Defina o método para configurar os elementos da imagem de origem
    def setup_src_image(self):
        global src_image, src_image_gray, src_mask, src_landmark_points, src_np_points, src_convexHull, indexes_triangles
        # Carregando a imagem de origem
        src_image = cv2.imread(self.src_image_index)
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

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Dicionário de marcadores do OpenCV
        dictonary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_250)
        parametros_markers = cv2.aruco.DetectorParameters_create()

        # Ajustar parâmetros do detector ArUco
        # parametros_markers.cornerRefinementMethod = cv.aruco.CORNER_REFINE_SUBPIX
        # parametros_markers.cornerRefinementWinSize = 5
        # parametros_markers.cornerRefinementMaxIterations = 60
        parametros_markers.cornerRefinementMinAccuracy = 0.1

        img = frame.to_ndarray(format="bgr24")

        # Atualizando as variáveis globais da imagem de origem
        global src_image, src_image_gray, src_mask, src_landmark_points, src_np_points, src_convexHull, indexes_triangles

        # Converter para cinza
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Aplicar filtros morfológicos
        gray = cv2.medianBlur(gray, 3)

        # Binarização da imagem
        # A intensidade do pixel é definida como 0, para toda a intensidade dos pixels, menor que o valor limite
        _, gray = cv2.threshold(gray, 10, 255, cv2.THRESH_TOZERO)
        # gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # Detecção ARUCO
        if TYPE in ('VIDEO', 'PORTAL'):
            corners, marker_ids, rej_corners = cv2.aruco.detectMarkers(gray, dictonary)

            if str(marker_ids) != "None":
                # Desenha os marcadores detectados na imagem
                cv2.aruco.drawDetectedMarkers(
                    image=img, corners=corners, borderColor=(0, 255, 0)
                )

                if TYPE == "VIDEO":
                    for ids, corners in zip(marker_ids, corners):
                        # Calcular o valor do índice para cada vídeo
                        index = (
                            config.TAGS_ID.index(ids[0]) if ids[0] in config.TAGS_ID else -1
                        )
                        if index >= 0:
                            img = ar_video.process_video_detection(
                                cv2,
                                np,
                                index,
                                self.demo,
                                self.detection_demo,
                                self.frame_count_demo,
                                self.aug_demo,
                                corners,
                                ids,
                                img,
                            )

                if TYPE == "PORTAL":
                    for ids, corners in zip(marker_ids, corners):
                        try:
                            img = ar_image.arucoAug(cv2, np, corners, ids, img, self.imagAugPortal)
                        except:
                            pass
        
        # DeepFake
        if TYPE == "Eu, Zequinha":
            try:
                # Obter os ponto de facias de referência da imagem de destino
                dest_landmark_points = media_utils.get_landmark_points(img)
                dest_np_points = np.array(dest_landmark_points)

                # Calcular a convexHull dos pontos de referência da imagem de destino
                dest_convexHull = cv2.convexHull(dest_np_points)

                # Obter as dimensões da imagem de destino
                height, width, channels = img.shape

                # Crtiar uma imagem vazia para a nova face
                new_face = np.zeros((height, width, channels), np.uint8)

                # Realizar a triangulação de ambos os rostos
                for triangle_index in indexes_triangles:
                    # Triangulação do primeiro rosto (imagem de origem)
                    points, src_cropped_triangle, cropped_triangle_mask, _ = media_utils.triangulation(
                        triangle_index=triangle_index,
                        landmark_points=src_landmark_points,
                        img=src_image
                    )

                    # Triangulação do segundo rosto (imagem de destino)
                    points2, _, dest_cropped_triangle_mask, rect = media_utils.triangulation(
                        triangle_index=triangle_index,
                        landmark_points=dest_landmark_points
                    )

                    # Aplicar a transformação de warp(transformação de perspectiva) sob as triangulações
                    warped_triangle = media_utils.warp_triangle(
                        rect=rect,
                        points1=points,
                        points2=points2,
                        src_cropped_triangle=src_cropped_triangle,
                        dest_cropped_triangle_mask=dest_cropped_triangle_mask
                    )

                    # Reconstruir o rosto de destino
                    media_utils.add_piece_of_new_face(
                        new_face=new_face, 
                        rect=rect, 
                        warped_triangle=warped_triangle
                    )
                
                # Realizar a substituição de rostos (face swapping)
                img = media_utils.swap_new_face(
                    dest_image=img, 
                    dest_image_gray=gray,
                    dest_convexHull=dest_convexHull,
                    new_face=new_face
                )

                # Aplicar um filtro de mediana para suavizar a imagem resultante
                img = cv2.medianBlur(img, 3)
                
            except:
                pass


        return av.VideoFrame.from_ndarray(img, format="bgr24")


# Configuração do aplicativo Streamlit
def main():
    webrtc_streamer(
        key="self",
        sendback_audio=False,
        video_transformer_factory=VideoTransformer,
        async_transform=True,
        media_stream_constraints={"video": True, "audio": False},
    ) 

    st.markdown("<h1 style='text-align: center;'>🕳️ AR Journey</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Hotel 10</h2>", unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.warning('O APP apresentou problemas ao ser carregado. Por favor, atualize a página.', icon='⚠️')

