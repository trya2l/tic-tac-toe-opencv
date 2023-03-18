
import numpy as np

# It creates a grid, adds symbols to the grid, checks if the grid is full, checks if there is a
# winner, and checks if the game is over


class tictactoe:

    def __init__(self, nb_lines, nb_columns):
        """
        The constructor of the class tictactoe

        :param nb_lines: number of lines in the grid
        :param nb_columns: number of columns in the grid
        """
        self.nb_lines = nb_lines
        self.nb_columns = nb_columns
        self.grid = np.zeros((nb_lines, nb_columns))
        self.winner = None
        self.game_over = False
        self.turn = 0

    def show_grid(self):
        """
        The function takes a grid as an argument and prints it

        :param grid: a 2D array of integers, where 0 represents an empty space, 1 represents a wall, and
        2 represents a goal
        """
        print(self.grid)

    def add_symbol(self, line, column, symbol):
        """
        It takes a grid, a line, a column and a symbol as arguments and returns the grid with the symbol
        added at the specified line and column

        :param grid: the grid that the player is playing on
        :param line: the line where the symbol will be added
        :param column: the column number of the grid
        :param symbol: the symbol to be added to the grid
        :return: The grid with the symbol added.
        """
        self.grid[line][column] = symbol

    def is_full(self, grid):
        """
        It returns True if the grid is full, and False if it isn't

        :param grid: the grid to check
        :return: The function is_full() is returning a boolean value.
        """
        for line in grid:
            for column in line:
                if column == 0:
                    return False
        return True

    def is_winner(self, grid, symbol):
        """
        It checks if the symbol is in all the rows, columns, and diagonals of the grid

        :param grid: The grid to check
        :param symbol: The symbol that the player is using
        :return: The function is_winner() returns True if the symbol is a winner, and False otherwise.
        """
        # Vérifie les lignes
        for line in grid:
            if np.all(line == symbol):
                return True

        # Vérifie les colonnes
        for column in grid.T:
            if np.all(column == symbol):
                return True

        # Vérifie les diagonales
        if np.all(np.diag(grid) == symbol):
            return True
        if np.all(np.diag(grid[::-1]) == symbol):
            return True

        return False

    def is_end(self):
        """
        If the grid is full or if there is a winner, then the game is over

        :param grid: a 2D array representing the current state of the game
        :return: The function is_end() returns a boolean value.
        """
        if self.is_full(self.grid):
            self.game_over = True
            self.winner = 0
            return True
        elif self.is_winner(self.grid, 1):
            self.game_over = True
            self.winner = 1
            return True
        elif self.is_winner(self.grid, 2):
            self.game_over = True
            self.winner = 2
            return True
        return False


    def computer_symbol(self):
        """
        The computer chooses a symbol

        :return: The function computer_symbol() returns an integer value.
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

    # a qui de jouer
    def who_play(self):
        """
        It returns the symbol of the player who is playing

        :return: The function who_play() returns an integer value.
        """
        if self.turn % 2 == 0:
            return 1
        else:
            return 2

    def who_won(self):
        """
        It returns the symbol of the winner

        :return: The function who_won() returns an integer value.
        """
        if self.winner == 1:
            return "Player 1"
        elif self.winner == 2:
            return "Player 2"
        else:
            return "Nobody"

    def computer_turn(self, symbol):
        """
        The computer plays a turn

        :return: the grid with the computer's symbol added
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
        """
        It checks if the case is empty

        :param line: the line of the case
        :param column: the column of the case
        :return: True if the case is empty, and False otherwise
        """
        if self.grid[line][column] == 0:
            return True
        else:
            return False

    def player_turn(self, line, column):
        """
        The player plays a turn

        :param line: the line of the case
        :param column: the column of the case
        :return: the grid with the player's symbol added
        """
        symbol = int(self.who_play())
        if self.check_case(line, column):
            self.grid[line][column] = symbol
            self.turn += 1
        else:
            print("This case is not empty")

        return self.grid
