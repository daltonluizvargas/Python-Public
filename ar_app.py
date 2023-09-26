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
    config_local as config,
)

st.set_page_config(
    page_title="AR Experience",
    page_icon="🧊",
)

# Constantes para os caminhos dos vídeos
TYPE = st.selectbox("Escolha o modo de AR: ", config.TYPE)
VIDEOS = [f"{config.PATH_VIDEOS}{id}.mp4" for id in config.TAGS_ID]
qtd_videos = len(VIDEOS)


# %%
class VideoTransformer(VideoProcessorBase):
    def __init__(self):
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

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# Configuração do aplicativo Streamlit
def main():

    # Configuração do streamlit_webrtc
    RTC_CONFIGURATION = RTCConfiguration(
        {
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}],
        }
    )

    VIDEO_CONSTRAINTS = {
    "width": {"ideal": 1280, "min": 800},
    "height": {"ideal": 720, "min": 600},
}

    webrtc_streamer(
        key="self",
        sendback_audio=False,
        video_transformer_factory=VideoTransformer,
        rtc_configuration=RTC_CONFIGURATION,
        async_transform=True,
        media_stream_constraints={"video": VIDEO_CONSTRAINTS, "audio": False},
    )

    st.title("Hotel 10")
    st.subheader("👁️‍🗨️ AR Experience")


if __name__ == "__main__":
    main()
