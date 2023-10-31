import cv2
import numpy as np
import imutils
import pytesseract
from imutils.video import VideoStream

rtsp_url = 'rtsp://user:password@ip:port/' 

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
SAVE_IMAGE = True
IMAGE_DIR = "./result/out"


def save_frame(frame, file_name, flip=True):
    # mudar de RGB para BGR
    if flip:
        cv2.imwrite(file_name, np.flip(frame, 2))
    else:
        cv2.imwrite(file_name, frame)


def detect_plate(img):
    (H, W) = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blur = cv2.bilateralFilter(gray, 9, 17, 17)
    # edged = cv2.Canny(blur, 30, 200)

    # BLACKHAT
    rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)

    # TRANSF. BINÁRIA
    squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKern)
    light = cv2.threshold(
        light, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # SOBEL
    gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F,
                      dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
    gradX = gradX.astype("uint8")
    gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)

    thresh = cv2.threshold(
        gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    thresh = cv2.bitwise_and(thresh, thresh, mask=light)
    thresh = cv2.dilate(thresh, None, iterations=2)
    thresh = cv2.erode(thresh, None, iterations=1)

    cv2.imshow('edged', thresh)
    conts = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    conts = imutils.grab_contours(conts)
    conts = sorted(conts, key=cv2.contourArea, reverse=True)[:5]

    location = None
    for c in conts:
        peri = cv2.arcLength(c, True)
        aprox = cv2.approxPolyDP(c, 0.02 * peri, True)
        if cv2.isContourConvex(aprox):
            if len(aprox) == 4:
                location = aprox
                break

    beginX = beginY = endX = endY = None
    if location is None:
        plate = False
    else:
        mask = np.zeros(gray.shape, np.uint8)

        img_plate = cv2.drawContours(mask, [location], 0, 255, -1)
        img_plate = cv2.bitwise_and(img, img, mask=mask)

        (y, x) = np.where(mask == 255)
        (beginX, beginY) = (np.min(x), np.min(y))
        (endX, endY) = (np.max(x), np.max(y))

        plate = gray[beginY:endY, beginX:endX]
        cv2.imshow('plate', plate)

    return img, plate, beginX, beginY, endX, endY


def ocr_plate(plate):
    text = pytesseract.image_to_string(plate, lang="por")
    text = "".join(c for c in text if c.isalnum())

    return text, valida_placa(text)


def valida_placa(placa):
    pos_letras = [0, 1, 2, 4]
    pos_numeros = [3, 5, 6]
    cont_letras = sum(1 for i, char in enumerate(
        placa) if i in pos_letras and char.isalpha())
    cont_numeros = sum(1 for i, char in enumerate(
        placa) if i in pos_numeros and char.isdigit())
    placa_valida = (cont_letras == 3 and cont_numeros == 4) or (
        cont_letras == 4 and cont_numeros == 3)
    if placa_valida:
        return True
    else:
        return False


def recognize_plate():
    vs = VideoStream(rtsp_url).start()
    while True:
        try:
            frame = vs.read()
            if frame is None:
                continue

            img, plate, beginX, beginY, endX, endY = detect_plate(frame)

            if plate is False:
                print("Não foi possível detectar a placa!")
            else:
                text, valida = ocr_plate(plate)
                if valida:
                    print('aqui')
                    if SAVE_IMAGE:
                        save_frame(img, IMAGE_DIR + "/vehicle_%04s.png" %
                                   text, flip=False)  # Salvar imagem do veículo
                        save_frame(plate, IMAGE_DIR + "/plate_%04s.png" %
                                   text, flip=False)  # Salvar placa detectada
                    img = cv2.putText(img, text, (beginX, beginY - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.9, (150, 255, 0), 2, lineType=cv2.LINE_AA)
                    img = cv2.rectangle(img, (beginX, beginY),
                                        (endX, endY), (150, 255, 0), 2)

            cv2.imshow('Imagem', img)

            # Add a delay in the loop to avoid high CPU usage
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        except KeyboardInterrupt:
            # Liberar o vídeo da memória e encerrar todas as janelas abertas
            cv2.destroyAllWindows()
            break


recognize_plate()
