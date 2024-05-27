import numpy as np
from scipy import linalg as la

from .config import TOL


def polyhedron_from_halfspaces(A, b):
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

    return np.array(intersections).T
