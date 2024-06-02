from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt

from ..mesh import Mesh, split_hyperface


vertices = np.array(
    [
        [0.0, 0.0, 0.0],
        [6.0, 0.0, 0.0],
        [0.0, 6.0, 0.0],
    ]
)
hyperfaces = np.array(
    [
        [0, 1, 2],
    ]
)
hyperface_norms = np.array(
    [
        [0, 0, 1],
    ],
    dtype=np.float64,
)

mesh = Mesh(vertices, hyperfaces, hyperface_norms)

h_vertices = np.array(
    [
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [4.0, 0.0, 0.0],
        [5.0, 1.0, 0.0],
        # [0.5, 2.0, 0.0],
        [0.0, 2.0, 0.0],
        [1.0, 2.0, 0.0],
        [1.0, 3.0, 0.0],
        [2.0, 2.0, 0.0],
        [3.0, 3.0, 0.0],
        # [0.5, 4.0, 0.0],
        [0.0, 4.0, 0.0],
        [1.0, 4.0, 0.0],
        [1.0, 5.0, 0.0],
    ]
)
h_edges = np.array(
    [
        [0, 1],
        [1, 2],
        [3, 4],
        [5, 6],
        [6, 7],
        [7, 8],
        [8, 9],
        [10, 11],
        [11, 12],
        # [5, 10],
    ]
)
h_edge_norms = np.array(
    [
        [-1, 0, 0],
        [0, -1, 0],
        [1, -1, 0],
        [0, 1, 0],
        [-1, 0, 0],
        [1, 1, 0],
        [-1, 1, 0],
        [0, -1, 0],
        [1, 0, 0],
        [1, 0, 0],
    ],
    dtype=np.float64,
)
h_edge_norms /= np.linalg.norm(h_edge_norms, axis=1, keepdims=True)

hole_cuts = (h_vertices, h_edges, h_edge_norms)

# h_vertices, h_edges = mesh.split_hyperface(0, hole_cuts)
vertices, new_hyperfaces = split_hyperface(mesh.vertices, mesh.hyperfaces[0], hole_cuts)
print(h_vertices, new_hyperfaces)


fig = plt.figure()
ax = plt.axes(projection="3d")

# for edge in hyperfaces:
for edge in new_hyperfaces:
    verts = vertices[edge]
    verts = np.vstack((verts, verts[0:1, :]))
    ax.plot(verts[:, 0], verts[:, 1], verts[:, 2], color="blue")

# for edge in h_edges:
#     verts = h_vertices[edge]
#     ax.plot(verts[:, 0], verts[:, 1], verts[:, 2], color="red")

plt.show()
