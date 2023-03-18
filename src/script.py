import numpy as np
import cv2
import matplotlib.pyplot as plt
import utils as u
import sys

import importlib
importlib.reload(u)

def main(argv):
    default_image = "img/image.png"
    filepath = argv[0] if len(argv) > 0 else default_image
    filename = argv[1] if len(argv) > 1 else "image"
    img = u.read_and_resize(800, filepath)

    plt.figure()
    plt.subplot(2, 2, 1)
    u.show(img)

    # Érosion

    kernel = 3
    element = cv2.getStructuringElement(
        cv2.MORPH_RECT, (2*kernel + 1, 2*kernel + 1), (kernel, kernel))

    erode = cv2.erode(img, element)

    plt.figure()
    plt.subplot(2, 2, 1)
    u.show(erode)

    # Median blur

    gray = cv2.cvtColor(erode, cv2.COLOR_BGR2GRAY)

    median = cv2.medianBlur(gray, 11)

    plt.figure()
    plt.subplot(2, 2, 1)
    u.show(median)

    # Canny

    canny = cv2.Canny(median, 100, 200)

    plt.figure()
    plt.subplot(2, 2, 1)
    u.show(canny)

    # Dilatation

    kernel = 0
    element = cv2.getStructuringElement(
        cv2.MORPH_RECT, (2*kernel + 1, 2*kernel + 1), (kernel, kernel))

    dilate = cv2.dilate(canny, element)

    plt.figure()
    plt.subplot(2, 2, 1)
    u.show(dilate)

    # Grid recogniton

    lines, corners = u.locate_grid(dilate, gray)

    plt.figure()
    plt.subplot(2, 2, 1)
    u.show(lines)

    # Grid rotation

    rotate, corners_t, img_rotate = u.rotate(lines, corners, img)

    plt.figure()
    plt.subplot(2, 2, 1)
    u.show(rotate)

    # Zoning

    zones = u.zoning(corners_t, img_rotate)

    path = u.export(zones, filename)

    resultat = u.symbols(path)
    print(resultat)

    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
