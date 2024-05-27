import numpy as np

from convex import polyhedron_from_halfspaces
from space import KSimplexSpace, nd_rotation

"""
To find the intersection of two simplexes, we first find the intersection space of the
two spaces that contain them. Then, we restrict the intersection space by the boundaries
of the two simplexes.
"""

T1 = np.random.rand(4, 4)
T2 = np.random.rand(4, 4)

# T1 = np.array(
#     [
#         [0, 0, 0, 0],
#         [-0.65, 0, 0, 0.4],
#         [0, 2.12, 0, -3.12],
#         [0, 0, 3.56, -1.34],
#     ]
# )
# t = 1.6
# RM = nd_rotation(t, 4, 0, 3)
# T2 = RM @ T1 + np.array([[0, 0.5, 1, 0]]).T
# T2 = T1 + np.array([[1, 1, 1, 1]]).T

# T1 = np.array(
#     [
#         [-1, 1, 1],
#         [0, 1, -1],
#         [0, 0, 0],
#     ]
# )
# t = 0.5
# RM = nd_rotation(t, 3, 0, 2)
# T2 = RM @ T1 + np.array([[0, 0, 0]]).T

S1 = KSimplexSpace(T1)
S2 = KSimplexSpace(T2)
# The intersection space of the two spaces
Si = KSimplexSpace.space_intersect(S1, S2)

# The dimension of the intersection space is one less than the dimension of the spaces
if Si.k == Si.dim - 2:
    # Find all constraints of the intersection space
    pA1, pb1 = S1.restrict_subspace(Si)
    pA2, pb2 = S2.restrict_subspace(Si)

    pA = np.vstack((pA1, pA2))
    pb = np.vstack((pb1, pb2))
    # print(np.hstack((pA, pb)))

    # Find the polyhedron by the constraints
    intersections_p = polyhedron_from_halfspaces(pA, pb)
    print(intersections_p.shape)

    import matplotlib.pyplot as plt

    fig = plt.figure()

    if intersections_p.shape[0] > 0:
        intersections = Si.O + Si.V @ intersections_p
        print(intersections)
        plt.plot(intersections_p[0, :], intersections_p[1, :], "o")

    x = np.array([-5, 5])
    for i in range(pA.shape[0]):
        y = (-pA[i, 0] * x + pb[i]) / pA[i, 1]
        plt.plot(x, y)

    plt.show()
