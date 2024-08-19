import numpy as np

# Example vectors v1 and v2
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# Normalize v1 and v2
v1_norm = v1 / np.linalg.norm(v1)
v2_norm = v2 / np.linalg.norm(v2)

# Example matrix M with shape (num_vectors, n)
M = np.array([[1, 0, 3],
              [7, 8, 9],
              [4, 2, 1]])

# Project each vector in M onto v1_norm and v2_norm
proj_v1 = (M @ v1_norm)[:, np.newaxis] * v1_norm  # Project onto v1, reshape to (n, 1) and broadcast
proj_v2 = (M @ v2_norm)[:, np.newaxis] * v2_norm  # Project onto v2, reshape to (n, 1) and broadcast

# Sum the projections to get the projection onto the plane spanned by v1 and v2
proj_plane = proj_v1 + proj_v2

print("Projections onto the plane spanned by v1 and v2:")
print(proj_plane)
