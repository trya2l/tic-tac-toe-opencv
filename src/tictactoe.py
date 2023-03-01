
import numpy as np

# It creates a grid, adds symbols to the grid, checks if the grid is full, checks if there is a
# winner, and checks if the game is over


class tictactoe:

    grid = None

    def __init__(self, nb_lines, nb_columns):
        """
        The constructor of the class tictactoe

        :param nb_lines: number of lines in the grid
        :param nb_columns: number of columns in the grid
        """
        self.grid = self.create_grid(nb_lines, nb_columns)



    def show_grid(self, grid):
        """
        The function takes a grid as an argument and prints it

        :param grid: a 2D array of integers, where 0 represents an empty space, 1 represents a wall, and
        2 represents a goal
        """
        print(grid)
        
    def add_symbol(self, grid, line, column, symbol):
        """
        It takes a grid, a line, a column and a symbol as arguments and returns the grid with the symbol
        added at the specified line and column

        :param grid: the grid that the player is playing on
        :param line: the line where the symbol will be added
        :param column: the column number of the grid
        :param symbol: the symbol to be added to the grid
        :return: The grid with the symbol added.
        """
        grid[line][column] = symbol
        return grid

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

    def is_end(self, grid):
        """
        If the grid is full or if there is a winner, then the game is over

        :param grid: a 2D array representing the current state of the game
        :return: The function is_end() returns a boolean value.
        """
        if self.is_full(grid) or self.is_winner(grid, 1) or self.is_winner(grid, 2):
            return True
        return False

    def computer_symbol(self, grid):
        """
        If there are more 1's than 2's in the grid, return 2, otherwise return 1

        :param grid: the grid of the game
        :return: the symbol of the computer.
        """
        nb_1 = 0
        nb_2 = 0
        for line in grid:
            for column in line:
                if column == 1:
                    nb_1 += 1
                elif column == 2:
                    nb_2 += 1
        if nb_1 > nb_2:
            return 2
        else:
            return 1

    """ def draw_symbol(self, grid, line, column, symbol):
    
        if symbol == 1:
            grid[line][column] = "X"
        elif symbol == 2:
            grid[line][column] = "O"
        return grid """
    
    #draw on the image
