import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg as la

from ..mesh import Mesh
from ..plot import plot_3D_mesh


def init_sculpt():
    # Create initial 4D cube
    vertices = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )

    pattern = np.array(
        [
            [0, 1, 2, 4],
            [1, 2, 3, 7],
            [2, 4, 6, 7],
            [1, 4, 5, 7],
            [1, 2, 4, 7],
        ]
    )

    hyperfaces = np.ndarray((0, 4), dtype=int)
    face_normals = []
    for i in range(4):

        base = 2**i
        mod_pattern = pattern % base
        new_pattern = (pattern - mod_pattern) * 2 + mod_pattern
        new_faces = np.vstack((new_pattern, new_pattern + base))

        for new_face in new_faces:
            face_vertices = vertices[new_face]
            norm = la.null_space(
                np.vstack(
                    (
                        face_vertices[3] - face_vertices[0],
                        face_vertices[2] - face_vertices[0],
                        face_vertices[1] - face_vertices[0],
                    )
                )
            ).T[0]

            # determine direction
            from_center = face_vertices[0] - 0.5
            norm *= np.sign(np.dot(from_center, norm))

            face_normals.append(norm)

        hyperfaces = np.vstack((hyperfaces, new_faces))

    return Mesh(vertices, hyperfaces, np.array(face_normals))


sculpt = init_sculpt()
sculpt3d = sculpt.project_3d_from_4d(
    [
        # (np.pi / 4, 0, 3),
        # (np.pi / 4, 1, 3),
        # (np.pi / 4, 2, 3),
        # (np.pi / 4, 0, 1),
        # (np.pi / 4, 0, 2),
        # (np.pi / 4, 1, 2),
    ]
)

# plot
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.set_xlim([-0.2, 1.2])
ax.set_ylim([-0.2, 1.2])
ax.set_zlim([-0.2, 1.2])


plot_3D_mesh(ax, sculpt3d, plot_normal=True)
sculpt3d.reorder_faces()
sculpt3d.to_wavefront("mesh/result/plot_4d.obj")

plt.show()
