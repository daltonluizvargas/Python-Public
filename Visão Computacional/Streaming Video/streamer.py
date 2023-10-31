import base64
import cv2
import zmq
import imutils
from imutils.video import VideoStream

context = zmq.Context()
footage_socket = context.socket(zmq.PUB)
footage_socket.connect('tcp://ip:port')

type_video = input('[INFO] Informe a conexão (rtsp, http, webcam): ')
# Webcam
video_source = 0

if type_video == 'rtsp':
    # RTSP
    # login:user 
    # Senha: password
    # Ip: rtsp://ip
    # Porta: 
    video_source = 'rtsp://user:password@ip:port/'  

if type_video == 'http':
    # HTTP 
    video_source = 'http://ip:port/video'

# Abrir o stream RTSP
vs = VideoStream(video_source).start() 

print('[INFO] Opening flow...')
while True:
    try:
        # Ler cada frame
        frame = vs.read()

        if frame is None:
            continue
        # frame = cv2.resize(frame, (640, 480))  # resize the frame
        encoded, buffer = cv2.imencode('.jpg', frame)
        jpg_as_text = base64.b64encode(buffer)
        footage_socket.send(jpg_as_text)

    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print('[INFO] Closing flow....')
        break
