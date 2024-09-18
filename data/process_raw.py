import os
import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

cctv_all = Path("raw/cctv_footage/surveillance_cameras_all")
annotations = Path("raw/cctv_footage/annotations/all.txt")

img2annot = dict()
for annotations in np.loadtxt(annotations, dtype=str):
    img2annot[annotations[0]] = annotations[1:]


def draw_bbox(img, bbox):
    x1, y1, x2, y2, x3, y3, x4, y4 = map(int, bbox)
    h, w, c = img.shape
    cv2.rectangle(img, (x1*2, y1*2), (x3, y3), (0, 255, 0), 1)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


for img_path in os.listdir(cctv_all):
    cam_img = cv2.imread(str(cctv_all / img_path))
    img_name = img_path.split('.')[0]
    print(img_name, cam_img.shape, img2annot[img_name])
    draw_bbox(cam_img, bbox=img2annot[img_name])
    break

