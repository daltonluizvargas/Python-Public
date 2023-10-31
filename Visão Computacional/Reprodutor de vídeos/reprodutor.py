# from kivy.app import App
# from kivy.uix.videoplayer import VideoPlayer

# class MyApp(App):
#     def build(self):
#         player = VideoPlayer(source='A:\Fontes\Python\Reprodutor de vídeos\4.mp4', state = 'play')
#         return player
    
# if __name__ == '__main__':
#     MyApp().run()


import cv2
import numpy as np
from ffpyplayer.player import MediaPlayer

def getVideoSource(source, width, height):
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap

def main():
    sourcePath = "2.mp4"
    camera = getVideoSource(sourcePath, 720, 480)
    player = MediaPlayer(sourcePath)

    while True:
            
        ret, frame = camera.read()
        audio_frame, val = player.get_frame()

        if (ret == 0):
            print("End of video")
            break

        frame = cv2.resize(frame, (720, 480))
        cv2.imshow('Camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if val != 'eof' and audio_frame is not None:
            frame, t = audio_frame
            print("Frame:" + str(frame) + " T: " + str(t))

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()