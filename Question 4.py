def transpose_matrix(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    transpose = [[0 for _ in range(rows)] for _ in range(cols)]
    
    for i in range(rows):
        for j in range(cols):
            transpose[j][i] = matrix[i][j]
    
    return transpose

def input_matrix(rows, cols):
    matrix = []
    for i in range(rows):
        row = list(map(int, input(f"Enter row {i+1} of the matrix: ").split()))
        if len(row) != cols:
            raise ValueError("Incorrect number of columns in row")
        matrix.append(row)
    return matrix

rows = int(input("Enter the number of rows for the matrix: "))
cols = int(input("Enter the number of columns for the matrix: "))
matrix = input_matrix(rows, cols)

transpose = transpose_matrix(matrix)
print("Transpose of the matrix:")
for row in transpose:
    print(" ".join(map(str, row)))