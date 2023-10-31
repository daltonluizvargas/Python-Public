import cv2
import sys
import logging as log
import datetime as dt
from time import sleep

from imutils.video import VideoStream
rtsp_url = 'rtsp://user:passqord@ip:port/'
http_url = 'http://ip:port/video'

TIPO          = 'rtsp'
WEBCAM        = 0

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)

if TIPO == 'rtsp':
    vs = VideoStream(rtsp_url).start() 
if TIPO == 'webcam':
    vs = cv2.VideoCapture(WEBCAM)
if TIPO == 'http':
    vs = VideoStream(http_url).start()
anterior = 0

while True:
    if TIPO != 'webcam':
        frame = vs.read()
        if frame is None:
            continue
    else:
        ret, frame = vs.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.5,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))


    # Display the resulting frame
    cv2.imshow('Video', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)

# When everything is done, release the capture
cv2.destroyAllWindows()