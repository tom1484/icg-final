import numpy as np
from scipy import linalg as la

from .config import TOL
from .utils import (
    remap_vertex_indexes,
    remove_redundant_vertices,
    remove_vertices_by_pos,
)


def polyhedron_from_halfspaces(A, b):
    """
    This function finds the intersection of halfspaces defined by A * x <= b.
    The algorithm is brute force, just to test the correctness of the intersection.
    """
    # brute force
    N = A.shape[0]
    dim = A.shape[1]

    perms = [[0 for _ in range(dim)] for _ in range(int(N**dim))]
    for i, perm in enumerate(perms):
        for j in range(dim):
            perm[j] = i % N
            i = i // N

    # filter out permutations that are not valid
    perms = [sorted(perm) for perm in perms if len(set(perm)) == dim]
    # filter out permutations that are redundant
    perms = sorted(perms)
    perms = [perms[i] for i in range(len(perms)) if i == 0 or perms[i] != perms[i - 1]]

    points_on_plane = [[] for _ in range(N)]
    intersections = []
    for perm in perms:
        A_perm = A[perm, :]
        b_perm = b[perm, :]
        # find the intersection of the halfspaces
        rk = np.linalg.matrix_rank(A_perm)
        if rk < dim:
            continue

        intersection = la.solve(A_perm, b_perm)
        valid = True

        for i in range(N):
            if not (i in perm or np.all(A[i, :] @ intersection - b[i] <= 0)):
                valid = False
                break

        if not valid:
            continue

        intersection = intersection.T[0]

        for p in intersections:
            if np.all(np.abs(p - intersection) < TOL):
                valid = False
                break

        if not valid:
            continue

        intersections.append(intersection)
        for i in perm:
            points_on_plane[i].append(len(intersections) - 1)

    edges = []
    for i in range(N):
        if len(points_on_plane[i]) >= dim:
            edges.append(points_on_plane[i])

    intersections = np.array(intersections)
    if intersections.shape[0] == 0:
        return intersections, []

    intersections, remap = remove_redundant_vertices(intersections)
    edges = [remap_vertex_indexes(edge, remap) for edge in edges]
    edges = [np.unique(sorted(edge)) for edge in edges]

    return intersections, edges
