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

# Step 2: Calculate the pseudo-inverse of A
pseudo_inverse_A = np.linalg.pinv(A)

# Step 3: Calculate the model vector X
X = pseudo_inverse_A @ C

# Print the results
print(f"Model vector X (Cost of each product):\n{X.flatten()}")
