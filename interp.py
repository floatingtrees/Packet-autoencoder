import numpy as np

# Sample arrays
a = np.array([1, 2, 3])
b = np.array([10, 20, 30, 40, 50, 60, 70])

indices_a = np.linspace(0, len(b) - 1, len(a))

interpolated_values = np.interp(indices_a, np.arange(len(b)), b)

print("Array a:", a)
print("Array b:", b)
print("Interpolated values:", interpolated_values)
