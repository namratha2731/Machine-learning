import numpy as np

# Step 1: Define matrices A and C
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

C = np.array([
    [386],
    [289],
    [393],
    [110],
    [280],
    [167],
    [271],
    [274],
    [148],
    [198]
])

# Step 2: Dimensionality of the vector space
dimensionality = A.shape[1]

# Step 3: Number of vectors
num_vectors = A.shape[0]

# Step 4: Rank of matrix A
rank_A = np.linalg.matrix_rank(A)

# Step 5: Calculate the cost using pseudo-inverse
pseudo_inverse_A = np.linalg.pinv(A)
costs = pseudo_inverse_A @ C

# Print the results
print(f"Dimensionality of the vector space: {dimensionality}")
print(f"Number of vectors in the vector space: {num_vectors}")
print(f"Rank of matrix A: {rank_A}")
print(f"Cost of each product: {costs.flatten()}")
