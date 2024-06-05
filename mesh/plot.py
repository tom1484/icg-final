import matplotlib.pyplot as plt
import numpy as np

from .mesh import Mesh


# NOTE: Testing functions
def plot_3D_mesh(ax, mesh: Mesh, plot_normal=False, color="blue"):
    for idx in range(mesh.num_faces):
        verts = mesh.get_hyperface(idx)
        verts = np.hstack((verts, verts[:, 0:1]))
        ax.plot(verts[0], verts[1], verts[2], color=color)

        if plot_normal:
            center = np.mean(verts, axis=1)
            ax.quiver(
                center[0],
                center[1],
                center[2],
                mesh.face_normals[idx][0] * 0.2,
                mesh.face_normals[idx][1] * 0.2,
                mesh.face_normals[idx][2] * 0.2,
                color="red",
            )
