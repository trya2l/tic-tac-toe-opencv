
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random

# taille des images
IMG_SIZE = 50

# charger une image
def load_image(path):
    img = cv2.imread(path)
    #img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img

# afficher une image
def show_image(img):
    plt.imshow(img)
    plt.show()

# filtre de Canny
def canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    return edges


# Hough transform
def hough_transform(edges):
    delta = 1
    M = []
    for x in IMG_SIZE :
        for y in IMG_SIZE :
            for theta in range(0,180,delta) :
                rho = x*cos(theta) + y*sin(theta)
                M[rho,theta] = M[rho,theta] + 1
            




# Main
def main():
    image = load_image("C:/Users/benoi/Documents/GitHub/images-labyrinthe/src/tictactoe.jpg")
    show_image(image)


main()