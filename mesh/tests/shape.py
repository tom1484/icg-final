import numpy as np

from convex import polyhedron_from_halfspaces
from space import KSimplexSpace

max_inter = 0
for i in range(3000):
    T1 = np.random.rand(4, 4)
    T2 = np.random.rand(4, 4)

    S1 = KSimplexSpace(T1)
    S2 = KSimplexSpace(T2)

    Si = KSimplexSpace.space_intersect(S1, S2)
    if Si.k == Si.dim - 2:
        pA1, pb1 = S1.restrict_subspace(Si)
        pA2, pb2 = S2.restrict_subspace(Si)

        pA = np.vstack((pA1, pA2))
        pb = np.vstack((pb1, pb2))
        # print(np.hstack((pA, pb)))

        intersections_p = polyhedron_from_halfspaces(pA, pb)
        if len(intersections_p.shape) == 1:
            continue

        # print(intersections_p.shape)
        max_inter = max(max_inter, intersections_p.shape[1])

        import matplotlib.pyplot as plt

        if intersections_p.shape[1] > 3:
            fig = plt.figure()

            if intersections_p.shape[0] > 0:
                intersections = Si.O + Si.V @ intersections_p
                print(intersections)
                plt.plot(intersections_p[0, :], intersections_p[1, :], "o")

            x = np.array([-5, 5])
            for i in range(pA.shape[0]):
                y = (-pA[i, 0] * x + pb[i]) / pA[i, 1]
                plt.plot(x, y, color="black" if i < pA1.shape[0] else "red")

            plt.xlim(-2, 2)
            plt.ylim(-2, 2)
            plt.show()

print(max_inter)
