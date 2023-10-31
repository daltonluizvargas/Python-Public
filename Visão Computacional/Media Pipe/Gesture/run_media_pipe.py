import cv2
import time
import requests
from imutils.video import VideoStream
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

from modulos import speak

rtsp_url = 'rtsp://user:password@ip:port/'
http_url = 'http://ip:port/video'

TIPO          = 'rtsp'
WEBCAM        = 0
SENHA         = 2
THRESHOLD     = 0.1

if TIPO == 'rtsp':
    vs = VideoStream(rtsp_url).start() 
if TIPO == 'webcam':
    vs = cv2.VideoCapture(WEBCAM)
if TIPO == 'http':
    vs = VideoStream(http_url).start()

def enviar_comando():
    url = "http://ip/?status=7"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print(f"Comando enviado: {url}")
        else:
            print(f"Erro ao enviar comando: {url}")
    except requests.exceptions.RequestException as e:
        print(f"Erro de conexão: {e}")

def confere_senha(fingerCount):
   if fingerCount == SENHA:
      enviar_comando() # Comando para abrir a porta 
      time.sleep(5)   

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  
  while True:
    if TIPO != 'webcam':
        image = vs.read()
        if image is None:
            continue
    else:
        ret, image = vs.read()

    image = cv2.resize(image, (1024,720))

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)


    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    fingerCount = 0

    if results.multi_hand_landmarks:

      for hand_landmarks in results.multi_hand_landmarks:
        
        handIndex = results.multi_hand_landmarks.index(hand_landmarks)
        handLabel = results.multi_handedness[handIndex].classification[0].label
        
        handLandmarks = []
        
        for landmarks in hand_landmarks.landmark:
          handLandmarks.append([landmarks.x, landmarks.y])

        if handLandmarks[8][1] < (handLandmarks[5][1] - THRESHOLD):       #Index finger
          fingerCount = fingerCount+1
        if handLandmarks[12][1] < (handLandmarks[9][1] - THRESHOLD):     #Middle finger
          fingerCount = fingerCount+1
        if handLandmarks[16][1] < (handLandmarks[13][1] - THRESHOLD):     #Ring finger
          fingerCount = fingerCount+1
        if handLandmarks[20][1] < (handLandmarks[17][1] - THRESHOLD):     #Pinky
          fingerCount = fingerCount+1

        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

    cv2.putText(image, str(fingerCount), (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 10)
    confere_senha(fingerCount)

    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break   

