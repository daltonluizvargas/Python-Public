'''
comando para gerar ARUCO:

CUBO 1
python gerar_aruco.py --id 0 --type DICT_5X5_100 --output tags/DICT_5X5_100_id0.png
python gerar_aruco.py --id 1 --type DICT_5X5_100 --output tags/DICT_5X5_100_id1.png
python gerar_aruco.py --id 2 --type DICT_5X5_100 --output tags/DICT_5X5_100_id2.png
python gerar_aruco.py --id 3 --type DICT_5X5_100 --output tags/DICT_5X5_100_id3.png
python gerar_aruco.py --id 4 --type DICT_5X5_100 --output tags/DICT_5X5_100_id4.png
python gerar_aruco.py --id 5 --type DICT_5X5_100 --output tags/DICT_5X5_100_id5.png

CUBO 1 - Teste com o marker 4x4_1000
python gerar_aruco.py --id 0 --type DICT_4X4_1000 --output tags/DICT_4X4_1000_id0.png
python gerar_aruco.py --id 1 --type DICT_4X4_1000 --output tags/DICT_4X4_1000_id1.png
python gerar_aruco.py --id 2 --type DICT_4X4_1000 --output tags/DICT_4X4_1000_id2.png
python gerar_aruco.py --id 3 --type DICT_4X4_1000 --output tags/DICT_4X4_1000_id3.png
python gerar_aruco.py --id 4 --type DICT_4X4_1000 --output tags/DICT_4X4_1000_id4.png
python gerar_aruco.py --id 5 --type DICT_4X4_1000 --output tags/DICT_4X4_1000_id5.png

CUBO 1 - Teste com o marker DICT_APRILTAG_36h11
python gerar_aruco.py --id 0 --type DICT_APRILTAG_36h11 --output tags/DICT_APRILTAG_36h11_id0.png
python gerar_aruco.py --id 1 --type DICT_APRILTAG_36h11 --output tags/DICT_APRILTAG_36h11_id1.png
python gerar_aruco.py --id 2 --type DICT_APRILTAG_36h11 --output tags/DICT_APRILTAG_36h11_id2.png
python gerar_aruco.py --id 3 --type DICT_APRILTAG_36h11 --output tags/DICT_APRILTAG_36h11_id3.png
python gerar_aruco.py --id 4 --type DICT_APRILTAG_36h11 --output tags/DICT_APRILTAG_36h11_id4.png
python gerar_aruco.py --id 5 --type DICT_APRILTAG_36h11 --output tags/DICT_APRILTAG_36h11_id5.png

CUBO 1 - Teste com o marker DICT_5X5_50
python gerar_aruco.py --id 0 --type DICT_5X5_50 --output tags/DICT_5X5_50_id0.png
python gerar_aruco.py --id 1 --type DICT_5X5_50 --output tags/DICT_5X5_50_id1.png
python gerar_aruco.py --id 2 --type DICT_5X5_50 --output tags/DICT_5X5_50_id2.png
python gerar_aruco.py --id 3 --type DICT_5X5_50 --output tags/DICT_5X5_50_id3.png
python gerar_aruco.py --id 4 --type DICT_5X5_50 --output tags/DICT_5X5_50_id4.png
python gerar_aruco.py --id 5 --type DICT_5X5_50 --output tags/DICT_5X5_50_id5.png



CUBO 2
python gerar_aruco.py --id 6 --type DICT_5X5_100 --output tags/DICT_5X5_100_id6.png
python gerar_aruco.py --id 7 --type DICT_5X5_100 --output tags/DICT_5X5_100_id7.png
python gerar_aruco.py --id 8 --type DICT_5X5_100 --output tags/DICT_5X5_100_id8.png
python gerar_aruco.py --id 9 --type DICT_5X5_100 --output tags/DICT_5X5_100_id9.png
python gerar_aruco.py --id 10 --type DICT_5X5_100 --output tags/DICT_5X5_100_id10.png
python gerar_aruco.py --id 11 --type DICT_5X5_100 --output tags/DICT_5X5_100_id11.png

CUBO 3
python gerar_aruco.py --id 12 --type DICT_5X5_100 --output tags/DICT_5X5_100_id12.png
python gerar_aruco.py --id 13 --type DICT_5X5_100 --output tags/DICT_5X5_100_id13.png
python gerar_aruco.py --id 14 --type DICT_5X5_100 --output tags/DICT_5X5_100_id14.png
python gerar_aruco.py --id 15 --type DICT_5X5_100 --output tags/DICT_5X5_100_id15.png
python gerar_aruco.py --id 16 --type DICT_5X5_100 --output tags/DICT_5X5_100_id16.png
python gerar_aruco.py --id 17 --type DICT_5X5_100 --output tags/DICT_5X5_100_id17.png


'''

import numpy as np
import argparse
import cv2
import sys

# Dicionário de tipos de ArUCo suportados pelo OpenCV
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
        "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
        "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
        "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
        "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
        "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
        "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
        "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
        "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
        "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
        "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
        "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
        "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
        "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
        "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
        "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
        "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", required=True,
                    help="caminho para a imagem de saída com a marca ArUCo")
    ap.add_argument("-i", "--id", type=int, required=True,
                    help="ID da marca ArUCo a ser gerada")
    ap.add_argument("-t", "--type", type=str,
                    default="DICT_ARUCO_ORIGINAL",
                    help="tipo de marca ArUCo a ser gerada")
    args = vars(ap.parse_args())

    if ARUCO_DICT.get(args["type"], None) is None:
        print("[INFO] O tipo de marca ArUCo '{}' não é suportado".format(args["type"]))
        sys.exit(0)

    arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args["type"]])
    print("[INFO] Gerando marca ArUCo do tipo '{}' com ID '{}'".format(args["type"], args["id"]))
    tag = np.zeros((300, 300, 1), dtype="uint8")
    cv2.aruco.drawMarker(arucoDict, args["id"], 300, tag, 1)

    cv2.imwrite(args["output"], tag)
    cv2.imshow("Marca ArUCo", tag)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
