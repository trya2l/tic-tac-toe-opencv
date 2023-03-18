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
import time
from colorama import Fore, Style

importlib.reload(u)


def load_image(path):
    img = cv2.imread(path)
    # img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img


def show_image(img):
    plt.imshow(img)
    plt.show()


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

    clear()
    return symboles


def clear():

    # for windows
    if os.name == 'nt':
        _ = os.system('cls')

    # for mac and linux
    else:
        _ = os.system('clear')


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
    return divisors[mid]


def startGame(img, filename):
    resultat = detect_form(load_image(img), filename)

    grid = ttt.tictactoe(int(value_to_grid(len(resultat))),
                         int(value_to_grid(len(resultat))))
    for i in range(3):
        for j in range(3):
            if resultat[i*3+j] == "CROIX":
                grid.add_symbol(i, j, "1")
            elif resultat[i*3+j] == "ROND":
                grid.add_symbol(i, j, "2")

    symbol = grid.computer_symbol()
    while grid.is_end() == False:
        grid.show_grid()
        print(" ")
        print("Tour :", grid.turn)
        print("Symbole de l'ordinateur est : " +
              (Fore.RED + "X" + Style.RESET_ALL if symbol == "1" else Fore.BLUE + "O" + Style.RESET_ALL))
        
        if symbol == "1":
            print("Symbole du joueur est : " + Fore.BLUE + "O" + Style.RESET_ALL)
        else:
            print("Symbole du joueur est : " + Fore.RED + "X" + Style.RESET_ALL)

        print(" ")
        if grid.who_play() == 1:
            print("C'est au tour du joueur 1.")
            print("Saisissez la ligne et la colonne de votre coup.")

            while True:
                line_p = int(input("Ligne : "))
                column_p = int(input("Colonne : "))
                if line_p < 0 or line_p > grid.size - 1 or column_p < 0 or column_p > grid.size - 1:
                    print("Veuillez saisir une valeur entre 0 et " +
                          str(grid.size - 1) + ".")
                else:
                    if grid.check_case(line_p, column_p):
                        break
                    else:
                        print("Cette case est déjà prise.")

            grid.player_turn(line_p, column_p)
        elif grid.who_play() == 2:
            print("C'est au tour de l'ordinateur.")
            grid.computer_turn(symbol)
            time.sleep(2)

        clear()

    print(Fore.GREEN + "Fin de la partie." + Style.RESET_ALL)
    if (grid.winner == 0):
        print("Match nul.")
    else:

        if symbol == "1":
            if grid.winner == 1:
                print("L'ordinateur a gagné.")
            else:
                print("Le joueur 1 a gagné.")
        else:
            if grid.winner == 1:
                print("Le joueur 1 a gagné.")
            else:
                print("L'ordinateur a gagné.")
    grid.show_grid()


def main():

    path = "img/"

    files = os.listdir(path)

    print("Sélectionnez une image :")
    for i, file in enumerate(files):
        print(f"{i+1}. {file}")

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
