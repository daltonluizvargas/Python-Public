import time
import cv2 as cv
from moviepy.editor import VideoFileClip
import pygame


def carregar_apresentacao_opencv(video_path):
    cap = cv.VideoCapture(video_path)

    # Enquanto o vídeo estiver aberto
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Ajustar vídeo de aprensetação para FULL SCREEM
        cv.namedWindow("Video", cv.WINDOW_NORMAL)
        cv.setWindowProperty("Video", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

        time.sleep(0.03)
        cv.imshow("Video", frame)

        # Pressionar 'q' para encerrar
        key = cv.waitKey(1)
        if key == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()


def carregar_apresentacao_pygame(video_path):
    clip = VideoFileClip(video_path)
    clip.preview()

    pygame.quit()
