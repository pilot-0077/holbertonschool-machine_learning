#!/usr/bin/env python3
"""
101-pca.py
Visualize Iris dataset in 3D after PCA reduction.
"""

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np

lib = np.load("pca.npz")
data = lib["data"]
labels = lib["labels"]

data_means = np.mean(data, axis=0)
norm_data = data - data_means
_, _, Vh = np.linalg.svd(norm_data)
pca_data = np.matmul(norm_data, Vh[:3].T)

# Plot PCA data in 3D
fig = plt.figure(figsize=(6.4, 4.8))
ax = fig.add_subplot(111, projection="3d")

ax.set_title("PCA of Iris Dataset")
ax.set_xlabel("U1")
ax.set_ylabel("U2")
ax.set_zlabel("U3")

sc = ax.scatter(
    pca_data[:, 0],
    pca_data[:, 1],
    pca_data[:, 2],
    c=labels,
    cmap="plasma"
)

plt.savefig("iris_plot.png")
plt.show()
