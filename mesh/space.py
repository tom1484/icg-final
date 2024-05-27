from typing import List, Optional, Tuple

import numpy as np
from scipy import linalg as la

from .config import TOL


class KSimplexSpace:
    """
    K-simplex is a convex hull of K + 1 points in R^N.
    It can be represented by:
    - vertices: T = O + [0, v1, v2, ..., vk], where O is the origin and vi are the basis vectors
    - equations: A * x = b, where A = null(V).T, b = A * O

    If the space is bounded, the orthogonal simplex containing the boundary points is also stored.
    """

    def __init__(
        self,
        # given vertices
        T: Optional[np.ndarray] = None,
        # given basis and origin
        V: Optional[np.ndarray] = None,
        O: Optional[np.ndarray] = None,
        # given equations
        A: Optional[np.ndarray] = None,
        b: Optional[np.ndarray] = None,
        # bounded
        bounded: bool = True,
    ):
        if T is not None:
            self.T = T
            self.A, self.b = self.vertices_to_eqation(self.T)

            self.V = self.T[:, 1:] - self.T[:, :1]
            self.O = self.T[:, :1]

            self.dim = self.A.shape[1]
            self.k = self.T.shape[1] - 1

        elif V is not None and O is not None:
            self.V = V
            self.O = O

            self.T = np.hstack((O, V + O))
            self.A, self.b = self.vertices_to_eqation(self.T)

            self.dim = self.A.shape[1]
            self.k = self.T.shape[1] - 1

        elif A is not None and b is not None:
            self.A = A
            self.b = b
            self.T = self.eqation_to_vertices(self.A, self.b)

            self.V = self.T[:, 1:] - self.T[:, :1]
            self.O = self.T[:, :1]

            self.dim = self.A.shape[1]
            self.k = self.T.shape[1] - 1

        self.bounded = bounded
        if bounded:
            self.ortho_bounds = self.construct_ortho_bounds()

    @staticmethod
    def vertices_to_eqation(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert vertices to equations according to the following formula:
        A * x = b, where A = null(V).T, b = A * O
        """
        V = T[:, 1:].T - T[:, 0].T
        A = la.null_space(V).T
        b = A @ T[:, 0]

        return A, b

    @staticmethod
    def eqation_to_vertices(A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Convert equations to vertices according to the following formula:
        T = O + [0, V], where V = pinv(A) @ b

        If the solution is invalid, return an empty array.
        """
        # I forgot what the following code does
        # It checks if the solution is valid anyway
        LU_decomp = la.lu(A, permute_l=True)
        L, U = LU_decomp[0], LU_decomp[1]

        Az = np.all(abs(U) <= TOL, axis=1)
        bnz = (abs(L @ b) > TOL).T[0]

        invalid = np.any(np.logical_and(Az, bnz))
        if invalid:
            return np.array([[]])

        O = la.pinv(A) @ b
        V = la.null_space(A)
        T = np.hstack((O, V + O))

        return T

    @classmethod
    def space_intersect(
        cls, S1: "KSimplexSpace", S2: "KSimplexSpace"
    ) -> "KSimplexSpace":
        """
        Find the intersection of two K-simplex spaces.
        We assume that the two spaces are unbounded.

        For example, if S1 and S2 are two 2-simplex spaces in R^3,
        the intersection is a 1-simplex space in R^3, which is a line.
        """
        # The equations of the intersection are the combination of the equations of S1 and S2
        Ai = np.vstack((S1.A, S2.A))
        bi = np.vstack((S1.b, S2.b))

        return KSimplexSpace(A=Ai, b=bi, bounded=False)

    def construct_ortho_bounds(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Construct the orthogonal simplices containing exactly K boundary points.

        For example, if the simplex is a 2-simplex (a triangle) in R^3,
        the orthogonal simplices are 3 2-simplex spaces,
        which are the orthogonal planes on the boundary edges of the triangle.
        """
        bounds = []

        # This function is only valid for K = N - 1
        if self.k != self.dim - 1:
            return bounds

        for i in range(self.k + 1):
            # boundary of the simplex
            bound = np.hstack((self.T[:, :i], self.T[:, i + 1 :]))
            # find the orthogonal simplex containing these boundary points
            Vb = bound[:, 1:].T - bound[:, :1].T
            Ao = la.null_space(np.vstack((self.A, Vb))).T
            bo = Ao @ bound[:, :1]
            # polarity of the boundary
            d = -np.sign(Ao @ self.T[:, i : i + 1] - bo)
            Ao *= d
            bo *= d

            bounds.append((Ao, bo))

        return bounds

    def restrict_subspace(self, S: "KSimplexSpace") -> Tuple[np.ndarray, np.ndarray]:
        """
        Construct the constraints of the subspace S by the boundaries of current simplex.

        For example, given:
        - current simplex: 2-simplex (a triangle) in R^3 defined by 3 vertices
          (0, 0), (2, 0), (0, 2)
        - subspace: a 1-simplex space (a line) in R^3 defined by one origin and one basis vector
          O = (1, 0), v1 = (0, 1)
        The subspace can written as O + a1 * v1 where a1 is a scalar.
        Than the constraints calculated by this function are:
        - -1 * a1 <= 0
        - 1 * a1 <= 1
        So pA = [1, 1].T, pb = [0, 1].T
        """
        N = S.T.shape[1] - 1

        pA = np.zeros((self.k + 1, N))
        pb = np.zeros((self.k + 1, 1))

        for i, (Ao, bo) in enumerate(self.ortho_bounds):
            # constraints of the subspace
            pA[i, :] = Ao @ S.V
            pb[i, :] = bo - Ao @ S.O

        return pA, pb


def nd_rotation(t: float, dim: int, ax1: int, ax2) -> np.ndarray:
    RM = np.eye(dim)
    cos = np.cos(t)
    sin = (-1) ** (ax1 + ax2) * np.sin(t)
    RM[ax1, ax1] = cos
    RM[ax1, ax2] = -sin
    RM[ax2, ax1] = sin
    RM[ax2, ax2] = cos

    return RM
