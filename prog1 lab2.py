import numpy as np
 
# Data from the image
# Matrix A (product quantities)
A = np.array([
    [20, 6, 2],
    [16, 3, 6],
    [27, 6, 2],
    [19, 1, 2],
    [24, 4, 2],
    [22, 1, 5],
    [15, 4, 2],
    [18, 4, 2],
    [21, 1, 4],
    [16, 2, 4]
])
 
# Matrix C (Payments)
C = np.array([386, 289, 393, 110, 280, 167, 271, 274, 148, 198])
 
# 1. Dimensionality of the vector space (number of columns in A)
dimensionality = A.shape[1]
 
# 2. Number of vectors in this vector space (number of rows in A)
num_vectors = A.shape[0]
 
# 3. Rank of matrix A
rank_of_A = np.linalg.matrix_rank(A)
 
# 4. Using Pseudo-Inverse to find the cost of each product
pseudo_inverse_A = np.linalg.pinv(A)
cost_of_products = pseudo_inverse_A @ C
 
# Printing the results
print(f"Dimensionality of the Vector Space: {dimensionality}")
print(f"Number of Vectors in this Vector Space: {num_vectors}")
print(f"Rank of Matrix A: {rank_of_A}")
print(f"Cost of Each Product: {cost_of_products}")
 