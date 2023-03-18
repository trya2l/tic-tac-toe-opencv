import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import old_files.functions as f
import tictactoe as ttt
import utils as u
import importlib
import math
importlib.reload(u)

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


def detect_form(img, filename):
    """
    It takes an image, applies some filters, detects the grid, rotates the image, zones it and returns
    the symbols

    :param img: the image to be processed
    :return: The result of the function is a list of lists.
    """
    show_image(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure()
    plt.subplot(2, 2, 1)

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
    resultat = [s.strip() for s in resultat]
    resultat = [s.replace(" ", "") for s in resultat]
    symboles = []

    for r in resultat:
        symbole = r.split(":")[1].strip()
        symboles.append(symbole)

    print(symboles)

    return symboles


def value_to_grid(value):
    """
    It finds the two numbers that are closest to the square root of the input value, and returns them in
    ascending order

    :param value: the number of items to be displayed in the grid
    :return: the two numbers that are closest to the square root of the input value.
    """
    divisors = []
    for i in range(1, int(math.sqrt(value))+1):
        if value % i == 0:
            divisors.append(i)
            if i != value // i:
                divisors.append(value // i)
    divisors.sort()
    mid = len(divisors) // 2
    return divisors[mid]  # , divisors[-mid]


def startGame(img, filename):
    resultat = detect_form(load_image(img), filename)
    print(resultat)
    print(value_to_grid(len(resultat)))

    grid = ttt.tictactoe(int(value_to_grid(len(resultat))),
                         int(value_to_grid(len(resultat))))
    # affiche la grille
    grid.show_grid()
    # remplir la grille avec les symboles détectés
    for i in range(3):
        for j in range(3):
            # resultat[1] = 1,1
            # resultat[2] = 1,2
            if resultat[i*3+j] == "CROIX":
                grid.add_symbol(i, j, "1")
            elif resultat[i*3+j] == "ROND":
                grid.add_symbol(i, j, "2")

    grid.show_grid()
    symbol = grid.computer_symbol()
    while grid.is_end() == False:
        print("Tour :", grid.turn)
        print("Play : ", grid.who_play())
        if grid.who_play() == 1:
            print("C'est au tour du joueur 1.")
            line_p = int(input("Ligne : "))
            column_p = int(input("Colonne : "))
            # check if input are between 0 and 2
            while line_p < 0 or line_p > 2 or column_p < 0 or column_p > 2:
                print("Erreur de saisie.")
                line_p = int(input("Ligne : "))
                column_p = int(input("Colonne : "))

            grid.player_turn(line_p, column_p)
        elif grid.who_play() == 2:
            print("C'est au tour de l'oordinateur.")
            grid.computer_turn(symbol)

        grid.show_grid()

    print("Fin de la partie.")
    print("Le gagnant est : " + grid.who_won())
    grid.show_grid()


def main():
    # chemin du dossier contenant les images
    path = "img/"

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

    startGame(image_path, files[choix-1])


main()
