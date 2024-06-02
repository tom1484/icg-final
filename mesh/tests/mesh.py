import numpy as np

from ..convex import polyhedron_from_halfspaces
from ..space import KSimplexSpace, nd_rotation


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

    tetrahedrons_id = np.ndarray((0, 4), dtype=int)
    for i in range(4):
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


def generate_cuts(mesh, RM):
    cuts = []
    patterns = np.array([[0, 1, 2, 3], [3, 4, 5, 1], [1, 2, 3, 5]])

    for _, face in enumerate(mesh["faces"]):
        vertices = mesh["vertices"][face] @ RM.T

        vertices0 = np.pad(vertices, ((0, 0), (0, 1)), constant_values=-0.1)
        vertices1 = np.pad(vertices, ((0, 0), (0, 1)), constant_values=1.1)

        vertices = np.vstack((vertices0, vertices1))
        
        for pattern in patterns:
            tetrahedron = vertices[pattern].T
            cuts.append(tetrahedron)

    return cuts

# TODO: Revert the mesh for cutting
mesh = {
    "vertices": np.array(
        [
            [0.2, 0.2, 0.2],
            [0.8, 0.2, 0.2],
            [0.2, 0.8, 0.2],
            [0.8, 0.8, 0.2],
            [0.2, 0.2, 0.8],
            [0.8, 0.2, 0.8],
            [0.2, 0.8, 0.8],
            [0.8, 0.8, 0.8],
        ]
    ),
    "faces": np.array(
        [
            [0, 1, 3],
            [0, 2, 3],
            [0, 1, 5],
            [0, 4, 5],
            [0, 2, 6],
            [0, 4, 6],
            [1, 3, 7],
            [1, 5, 7],
            [2, 3, 7],
            [2, 6, 7],
            [4, 5, 7],
            [4, 6, 7],
        ]
    ),
}
RM = nd_rotation(0.7, 3, 0, 2)

cuts = generate_cuts(mesh, RM)

for T in cuts:
    S = KSimplexSpace(T)
    print(S.T)

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

# S1 = KSimplexSpace(T1)
# S2 = KSimplexSpace(T2)
#
# Si = KSimplexSpace.space_intersect(S1, S2)
# if Si.k == Si.dim - 2:
#     pA1, pb1 = S1.restrict_subspace(Si)
#     pA2, pb2 = S2.restrict_subspace(Si)
#
#     pA = np.vstack((pA1, pA2))
#     pb = np.vstack((pb1, pb2))
#     # print(np.hstack((pA, pb)))
#
#     intersections_p = polyhedron_from_halfspaces(pA, pb)
#     print(intersections_p.shape)
#
#     import matplotlib.pyplot as plt
#
#     fig = plt.figure()
#
#     if intersections_p.shape[0] > 0:
#         intersections = Si.O + Si.V @ intersections_p
#         print(intersections)
#         plt.plot(intersections_p[0, :], intersections_p[1, :], "o")
#
#     x = np.array([-5, 5])
#     for i in range(pA.shape[0]):
#         y = (-pA[i, 0] * x + pb[i]) / pA[i, 1]
#         plt.plot(x, y)
#
#     plt.show()
