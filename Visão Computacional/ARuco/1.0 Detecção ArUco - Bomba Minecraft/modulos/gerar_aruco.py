# %%
import numpy as np
import argparse
import cv2
import sys

"""
comando para gerar ARUCO:
python gerar_aruco.py --id 10 --type DICT_5X5_100 --output tags/DICT_5X5_100_id10.png
python gerar_aruco.py --id 24 --type DICT_5X5_100 --output tags/DICT_5X5_100_id24.png
python gerar_aruco.py --id 2 --type DICT_5X5_100 --output tags/DICT_5X5_100_id2.png
python gerar_aruco.py --id 7 --type DICT_5X5_100 --output tags/DICT_5X5_100_id7.png
python gerar_aruco.py --id 15 --type DICT_5X5_100 --output tags/DICT_5X5_100_id15.png
python gerar_aruco.py --id 32 --type DICT_5X5_100 --output tags/DICT_5X5_100_id32.png
"""

# define os nomes de cada marca ArUco possível que o OpenCV suporta
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

# constrói o analisador de argumentos e analisa os argumentos
ap = argparse.ArgumentParser()
ap.add_argument(
    "-o", "--output", required=True, help="path to output image containing ArUCo tag"
)
ap.add_argument(
    "-i", "--id", type=int, required=True, help="ID of ArUCo tag to generate"
)
ap.add_argument(
    "-t",
    "--type",
    type=str,
    default="DICT_ARUCO_ORIGINAL",
    help="type of ArUCo tag to generate",
)
args = vars(ap.parse_args())

# verifique se a tag ArUCo fornecida existe e é suportada por
# OpenCV
if ARUCO_DICT.get(args["type"], None) is None:
    print("[INFO] ArUCo tag of '{}' is not supported".format(args["type"]))
    sys.exit(0)

# carrega o dicionário ArUCo
arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args["type"]])

# aloca memória para a tag ArUCo de saída e então desenha o ArUCo
# tag na imagem de saída
print(
    "[INFO] generating ArUCo tag type '{}' with ID '{}'".format(
        args["type"], args["id"]
    )
)
tag = np.zeros((300, 300, 1), dtype="uint8")
cv2.aruco.drawMarker(arucoDict, args["id"], 300, tag, 1)

# grava o tag ArUCo gerado no disco e então o exibe em nosso
# tela
cv2.imwrite(args["output"], tag)
cv2.imshow("ArUCo Tag", tag)
cv2.waitKey(0)
