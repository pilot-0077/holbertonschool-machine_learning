#!/usr/bin/env python3

if __name__ == '__main__':
    determinant = __import__('0-determinant').determinant

    mat0 = [[]]
    mat1 = [[5]]
    mat2 = [[1, 2], [3, 4]]
    mat3 = [[1, 1], [1, 1]]
    mat4 = [[5, 7, 9], [3, 1, 8], [6, 2, 4]]
    mat5 = []
    mat6 = [[1, 2, 3], [4, 5, 6]]

    print(determinant(mat0))  # Expected: 1
    print(determinant(mat1))  # Expected: 5
    print(determinant(mat2))  # Expected: -2
    print(determinant(mat3))  # Expected: 0
    print(determinant(mat4))  # Expected: 192

    try:
        determinant(mat5)
    except Exception as e:
        print(e)  # Expected: matrix must be a list of lists

    try:
        determinant(mat6)
    except Exception as e:
        print(e)  # Expected: matrix must be a square matrix
