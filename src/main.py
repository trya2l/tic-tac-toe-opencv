import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import functions as f
import tictactoe as ttt

# Taille des images
#IMG_SIZE = 829


# Charger une image

def load_image(path):
    img = cv2.imread(path)
    #img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
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
    lignes_detected = []
    lignes = cv2.HoughLines(edges, 1, np.pi/180,
                            int(np.trunc(img.shape[0]/3.5)))    # diminuer le dénominateur diminue le seuil

    if lignes is None:
        print("No lines detected in the image")
        return

    for line in lignes:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)

        isSimilar = False
        for r, t in lignes_detected:
            if abs(r - rho) < 10 and abs(t - theta) < 0.1:
                isSimilar = True
                break
        if not isSimilar:
            lignes_detected.append((rho, theta))

            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    #print("Number of lines detected: ", len(lignes_detected))
    cv2.imwrite('houghlines3.jpg', img)
    cv2.imread('houghlines3.jpg')
    cv2.imshow('houghlines3.jpg', img)
    cv2.waitKey(0)

    startGame(img, lignes_detected)
    return lignes_detected


def startGame(img, lignes_detected):
    morpion = ttt.tictactoe(lignes_detected/2, lignes_detected/2)
    morpion.show_grid(morpion.grid)
    while(morpion.is_end(morpion.grid) == False):
        morpion.show_grid(morpion.grid)
        print("C'est au tour du joueur 1")
        line = int(input("Entrez la ligne : "))
        column = int(input("Entrez la colonne : "))
        morpion.grid = morpion.add_symbol(morpion.grid, line, column, 1)
        morpion.show_grid(morpion.grid)
        if morpion.is_end(morpion.grid) == True:
            break
        print("C'est au tour du joueur 2")
        line = int(input("Entrez la ligne : "))
        column = int(input("Entrez la colonne : "))
        morpion.grid = morpion.add_symbol(morpion.grid, line, column, 2)
    morpion.show_grid(morpion.grid)
    print("Fin de la partie")


# Reconnaissance de symboles, soit X soit O dans une case de la grille


def recognize_symbol(img):
    dim = img.size
    cv2.imshow('base', img)
    imgBase = img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img, 5)

    cercles = cv2.HoughCircles(
        img, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=int(dim/9-dim/5), maxRadius=int(dim/9+dim/5))

    cercles = np.uint16(np.around(cercles))
    for i in cercles[0, :]:
        cv2.circle(imgBase, (i[0], i[1]), i[2],
                   (0, 255, 0), 2)     # périmètre du cercle
        # centre du cercle
        cv2.circle(imgBase, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv2.imshow('cercles detectes', imgBase)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    # chemin du dossier contenant les images
    path = 'src/img/'

    # liste des fichiers dans le dossier
    files = os.listdir(path)

    # affichage du menu
    print("Sélectionnez une image :")
    for i, file in enumerate(files):
        print(f"{i+1}. {file}")

    # demande de saisie de l'utilisateur
    while True:
        try:
            choix = int(input("Votre choix : "))
            if choix < 1 or choix > len(files):
                raise ValueError
            break
        except ValueError:
            print("Veuillez entrer un choix valide.")

    # sélection de l'image choisie
    image_path = path + files[choix-1]

    print(f"Vous avez sélectionné l'image : {image_path}")

    image = load_image(image_path)

    edges = canny(image)
    # show_image(edges)
    hough_transform(image, edges)
    #recognize_symbol(image)
    #print(hough_transform(image, edges))


main()
