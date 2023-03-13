import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random


# Taille des images
# IMG_SIZE = 829


# Charger une image

def load_image(path):
    img = cv2.imread(path)
    # img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img

# Afficher une image


def show_image(img):
    plt.imshow(img)
    plt.show()

# Filtre de Canny : contours de l'img


def canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    return edges

# Transformee de Hough


def hough_transform(img, edges):
    """
    It takes an image and its edges, and applies the Hough transform to detect lines in the image

    :param img: The image where we want to draw the lines
    :param edges: Output of the edge detector. It should be a grayscale image (although in fact it is a
    binary one) and it should have the same size as image
    :return: the lines detected in the image.
    """
    lignes = cv2.HoughLines(edges, 1, np.pi/180, int(np.trunc(img.shape/4, 5)))

    if lignes is None:
        print("No lines detected in the image")
        return

    for line in lignes:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imwrite('houghlines3.jpg', img)
    cv2.imread('houghlines3.jpg')
    cv2.imshow('houghlines3.jpg', img)
    cv2.waitKey(0)

# Reconnaissance de symboles


def recognize_symbol(img):
    pass

# Reconnaissance fin de partie


def recognize_end(img):
    pass
