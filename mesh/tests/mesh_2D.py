import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d

from ..convex import polyhedron_from_halfspaces
from ..space import KSimplexSpace, nd_rotation


def init_sculpt():
    # Create initial 4D cube
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )

    pattern = np.array(
        [
            [0, 1, 3],
            [0, 2, 3],
        ]
    )

    tetrahedrons_id = np.ndarray((0, 3), dtype=int)
    for i in range(3):
        base = 2**i
        mod_pattern = pattern % base
        new_pattern = (pattern - mod_pattern) * 2 + mod_pattern

        tetrahedrons_id = np.vstack((tetrahedrons_id, new_pattern))
        tetrahedrons_id = np.vstack((tetrahedrons_id, new_pattern + base))

    return {
        "vertices": vertices,
        "tetrahedrons_id": tetrahedrons_id,
    }


def get_sculpt_tetrahedron(sculpt, index):
    tetrahedrons_id = sculpt["tetrahedrons_id"]
    return sculpt["vertices"][tetrahedrons_id[index]].T


sculpt = init_sculpt()
# print(sculpt["tetrahedrons_id"])


def generate_cuts(mesh, RM):
    cuts = []
    # pattern = np.array([[0, 1, 2, 3], [3, 4, 5, 1], [1, 2, 3, 5]])
    patterns = np.array([[0, 1, 2], [1, 2, 3]])

    for _, face in enumerate(mesh["faces"]):
        vertices = mesh["vertices"][face] @ RM.T

        vertices0 = np.pad(vertices, ((0, 0), (0, 1)), constant_values=0)
        vertices1 = np.pad(vertices, ((0, 0), (0, 1)), constant_values=1)

        vertices = np.vstack((vertices0, vertices1))

        for pattern in patterns:
            tetrahedron = vertices[pattern].T
            cuts.append(tetrahedron)

    return cuts


# TODO: Revert the mesh for cutting
mesh = {
    "vertices": np.array(
        [
            [0.2, 0.2],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.8, 0.8],
        ]
    ),
    "faces": np.array(
        [
            [0, 1],
            [0, 2],
            [1, 3],
            [2, 3],
        ]
    ),
}
# RM = nd_rotation(0.7, 2, 0, 1)
RM = np.identity(2)

cuts = generate_cuts(mesh, RM)


# Display
fig = plt.figure()
ax = plt.axes(projection="3d")

for cut in cuts:
    cut = np.hstack((cut, cut[:, 0:1]))
    ax.plot(cut[0], cut[1], cut[2], color="green")

for j in range(len(sculpt["tetrahedrons_id"])):
    T = get_sculpt_tetrahedron(sculpt, j)
    tetrahedron = np.hstack((T, T[:, 0:1]))
    ax.plot(tetrahedron[0], tetrahedron[1], tetrahedron[2], color="blue")

for T in cuts:
    S = KSimplexSpace(T)

    for j in range(len(sculpt["tetrahedrons_id"])):
        ST = KSimplexSpace(get_sculpt_tetrahedron(sculpt, j))
        Si = KSimplexSpace.space_intersect(S, ST)

        if Si.k == Si.dim - 2:
            pA1, pb1 = S.restrict_subspace(Si)
            pA2, pb2 = ST.restrict_subspace(Si)

            pA = np.vstack((pA1, pA2))
            pb = np.vstack((pb1, pb2))

            intersections_p = polyhedron_from_halfspaces(pA, pb)

            if intersections_p.shape[0] > 0:
                intersections = Si.O + Si.V @ intersections_p
                # print(ST.T)
                print(intersections)
                # ax.scatter(
                #     intersections[0], intersections[1], intersections[2], color="red"
                # )

            # TODO: Create new faces from intersections


# plt.show()
