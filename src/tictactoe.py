
import numpy as np
from colorama import Fore, Style

class tictactoe:

    def __init__(self, nb_lines, nb_columns):

        self.nb_lines = nb_lines
        self.nb_columns = nb_columns
        self.grid = np.zeros((nb_lines, nb_columns))
        self.winner = None
        self.game_over = False
        self.turn = 0
        self.size = nb_lines

    def __str__(self):
        return "Nombre de lignes : " + str(self.nb_lines) + " Nombre de colonnes : " + str(self.nb_columns) + " Gagnant : " + str(self.winner) + " Fin de la partie : " + str(self.game_over) + " Tour : " + str(self.turn)

    def show_grid(self):
        
        print("    0 1 2")
        print("   _______")
        for i in range(self.nb_lines):
            print(i, end=" | ")
            for j in range(self.nb_columns):
                if self.grid[i][j] == 0:
                    print("-", end=" ")
                elif self.grid[i][j] == 1:
                    print(Fore.RED + "X" + Style.RESET_ALL, end=" ")
                elif self.grid[i][j] == 2:
                    print(Fore.BLUE + "O" + Style.RESET_ALL, end=" ")
            print()

    def add_symbol(self, line, column, symbol):
        self.grid[line][column] = symbol

    def is_full(self):
        """
        It returns True if the grid is full, and False if it isn't

        :param grid: the grid to check
        :return: The function is_full() is returning a boolean value.
        """
        return np.all(self.grid != 0)
    

    def is_winner(self, symbol):
        """
        If any of the rows, columns, or diagonals are all the same symbol, then that symbol is the winner
        
        :param symbol: The symbol to check for a win
        :return: The function is_winner() returns True if the symbol is a winner, and False otherwise.
        """
        for line in self.grid:
            if np.all(line == symbol):
                return True

        for column in self.grid.T:
            if np.all(column == symbol):
                return True

        if np.all(np.diag(self.grid) == symbol):
            return True
        if np.all(np.diag(np.fliplr(self.grid)) == symbol):
            return True

        return False

    def is_end(self):
        if self.is_winner(1):
            self.game_over = True
            self.winner = 1
            return True
        elif self.is_winner(2):
            self.game_over = True
            self.winner = 2
            return True
        elif self.is_full():
            self.game_over = True
            self.winner = 0
            return True
        return False

    def computer_symbol(self):
        """
        It returns 1 if there are more 2's than 1's in the grid, and 2 otherwise
        :return: The number of the symbol that the computer will play with.
        """
        nb_1 = 0
        nb_2 = 0
        for line in self.grid:
            for column in line:
                if column == 1:
                    nb_1 += 1
                elif column == 2:
                    nb_2 += 1
        if nb_1 < nb_2:
            return 1
        else:
            return 2

    def who_play(self):
        if self.turn % 2 == 0:
            return 1
        else:
            return 2

    def computer_turn(self, symbol):
        """
        The computer chooses a random line and column, and if the cell is empty, it places its symbol
        there
        
        :param symbol: 1 or 2
        :return: The grid
        """
        line = np.random.randint(0, self.nb_lines)
        column = np.random.randint(0, self.nb_columns)
        while self.grid[line][column] != 0:
            line = np.random.randint(0, self.nb_lines)
            column = np.random.randint(0, self.nb_columns)
        self.grid[line][column] = symbol
        self.turn += 1
        return self.grid

    def check_case(self, line, column):
        if self.grid[line][column] == 0:
            return True
        else:
            return False

    def player_turn(self, line, column):
        symbol = int(self.who_play())
        self.grid[line][column] = symbol
        self.turn += 1
        return self.grid
