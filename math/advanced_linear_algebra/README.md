# Advanced Linear Algebra - Task 0: Determinant

## 🔍 Task Description
This task implements a function `determinant(matrix)` to calculate the determinant of a square matrix using recursive expansion by minors.

## ✅ Function Requirements
- Input must be a list of lists (square matrix)
- Raises `TypeError` if not a list of lists
- Raises `ValueError` if the matrix is not square
- 0x0 matrix returns `1` by convention
- Recursive computation for nxn matrices

## 💡 Sample Usage

```python
determinant([[]])               # Output: 1
determinant([[5]])              # Output: 5
determinant([[1, 2], [3, 4]])   # Output: -2
