"""Gaussian Elimination Program 1.0 -- No custom classes

The algorithm used to construct the program is based off the steps from 'Elimination Algorithm 2'
(pg 46, "A Course in Linear Algebra", Damiano & Little). Further perfecting may be tried in future
in order to take into account limitations of floating point arithmetic, numerical instability,
unstable functions, ect. Will also try to get a power method dominant eigenvalue function done in the near future.

Copyright (c) 2024 Lunes Maibeche

"""

from typing import List

# Example matrices and vectors used in doctests - note their entries can be expressed with floats or ints
EXAMPLE_SOE = [[1.0, 5.0], [1.0, 1.0]]

EXAMPLE_SOL = [6.0, 2.0]

EXAMPLE_AUG_2X2 = [[1.0, 5.0, '|', 6.0],
                   [1.0, 1.0, '|', 2.0]]

EXAMPLE_AUG_3X3 = [[1.0, 0.0, 2.0, '|', 12.0],
                   [0.0, 2.0, 1.0, '|', 11.0],
                   [1.0, 0.0, 1.0, '|', 7.0]]

# May be used to account for float arithmetic rounding errors (but isn't currently being used by any functions)
EPSILON = 0.005


# Function to create an augmented matrix from a system of equations based on coefficients (as a matrix) and
# a solutions vector.
def create_augmented_matrix(coefficients: list[list[int]], solutions: list) -> list:
    """Returns an augmented matrix given the coefficients for a system of linear equations associated with given
    variables, and the solutions associated with each linear equation.

    Preconditions: Length of coefficients list is the same as length of solutions list.

    >>>create_augmented_matrix(EXAMPLE_SOE, EXAMPLE_SOL)
    [[1.0, 5.0, '|', 6.0],[1.0, 1.0, '|', 2.0]]

    """

    dim = len(solutions)
    augmatrix = []

    for i in range(0, dim):
        row = []
        row.extend(coefficients[i])
        row.append(solutions[i])
        row.insert(-1, '|')
        augmatrix.append(row)
    return augmatrix


# Gaussian elimination subroutine helper functions
def addrows(matrix: list, row_2_index: int, row_1_index: int) -> None:
    """Adds a given row_2 to row_1 based on their indices, updating row_1's entries.

    >>> add_rows(EXAMPLE_AUG_3X3, 1, 0)
    >>> EXAMPLE_AUG_3X3
    [[1.0, 2.0, 3.0, '|', 23.0], [0.0, 2.0, 1.0, '|', 11.0], [1.0, 0.0, 1.0, '|', 7.0]]

    """

    i = 0
    for entry in matrix[row_1_index]:
        if not isinstance(entry, str):
            matrix[row_1_index][i] = entry + matrix[row_2_index][i]
        i += 1
    pass


def multiply_row(matrix: list, scalar: float, row_index: int) -> None:
    """Updates a given row by multiplying the row's entries by a given scalar.

    >>>multiply_row(EXAMPLE_AUG_2X2, 2.0, 0)
    >>> EXAMPLE_AUG_2X2
    [[2.0, 10.0, '|', 12.0], [1.0, 1.0, '|', 2.0]]

    """

    i = 0
    for entry in matrix[row_index]:
        if not isinstance(entry, str):
            matrix[row_index][i] = entry * scalar
        i += 1
    pass


def interchange_rows(matrix: list, row_1_index: int, row_2_index: int) -> None:
    """Interchanges the order of a row_1 and row_2 in a given matrix.

    >>> interchange_rows(EXAMPLE_AUG_3X3,0,1)
    >>>EXAMPLE_AUG_3X3
    [[0, 2, 1, '|', 11], [1, 0, 2, '|', 12], [1, 0, 1, '|', 7]]
    """

    row_1 = matrix[row_1_index]
    row_2 = matrix[row_2_index]
    matrix[row_1_index] = row_2
    matrix[row_2_index] = row_1
    pass


# Gaussian Elimination Algorithm doesn't work very well, uncertain why
def gaussian_elimination(matrix: list) -> list:
    """Executes the gaussian elimination algorithm on a given matrix and returns its equivalent echelon form.

    The elimination algorithm is as follows (As written in "A Course in Linear Algebra", Damiano & Little)

    1. For each i from 1 to m from the matrix's m equations:

    2. Pick one equation containing the variable with the smallest index among the variables occuring in those equations
        - If necessary interchange that equation with the ith equation using interchange
        - Make the leading coefficient of this new ith equation equal to zero using multiply by scalar
        - Eliminate occurences of the leading variable in all other equations numbered i+1 through m

    3. For each i from m to 2 in reverse order:
        - Eliminate the occurences of the leading variable in equation i in equations 1 through i - 1

    >>>gaussian_elimination(EXAMPLE_AUG_3X3)
    [[1.0, 0.0, 0.0, '|', 2.0], [0.0, 1.0, 0.0, '|', 3.0], [0.0, 0.0, 1.0, '|', 5.0]]

    """

    m = len(matrix) # Number of rows
    n = matrix[0].index('|')  # Number of columns
    for j in range(n):
        k = 0
    # Check rows
        while k < len(matrix)-1:
            if matrix[k][j] != 0:
                break
            k += 1
    # Swap row of leading coefficient (if exists, else skip to next column)
        interchange_rows(matrix, k, j)
    # Make leading coefficient equal to 1
        multiply_row(matrix, 1/matrix[j][j], j)
    # Cancel out occurrences of the leading variable in all other equations from i+1 to m
        for i in range(k+1, m):
            if matrix[i][j] != 0:
                add_rows(matrix, -1/matrix[i][j]*matrix[k], matrix[i])
    # Cancel out occurrences of the leading variable in all other equations from 1 to i-1
        for i in range(1,k-1):
            if matrix[i][j] != 0:
                add_rows(matrix, -1/matrix[i][j]*matrix[k], matrix[i])
    return matrix
