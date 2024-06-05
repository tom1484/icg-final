from typing import List, Tuple

import meshpy.triangle as triangle
import numpy as np
from scipy import linalg as la
from tqdm import tqdm

from .config import TOL


class Mesh:
    def __init__(
        self,
        vertices: np.ndarray,
        hyperfaces: np.ndarray,
        face_normals: np.ndarray,
    ):
        self.vertices = vertices
        self.hyperfaces = hyperfaces

        assert self.vertices.shape[1] == self.hyperfaces.shape[1]
        # if not empty
        if self.hyperfaces.shape[0] > 0:
            assert np.max(self.hyperfaces.flatten()) < self.vertices.shape[0]

        self.num_verts = self.vertices.shape[0]
        self.num_faces = self.hyperfaces.shape[0]

        norms = la.norm(face_normals, axis=1, keepdims=True)
        norms[norms < TOL] = 1  # pyright: ignore
        face_normals /= norms
        self.face_normals = face_normals

    @classmethod
    def from_wavefront(cls, filename: str):
        vertices = []
        faces = []
        with open(filename) as f:
            for line in f:
                if line[0] == "v":
                    vertex = list(map(float, line[2:].strip().split()))
                    vertices.append(vertex)
                elif line[0] == "f":
                    parse = [v_vn.split("//") for v_vn in line[2:].strip().split()]
                    face = [int(v) - 1 for v, _ in parse]
                    faces.append(face)

        return cls(np.array(vertices), np.array(faces), np.empty(0))

    def to_wavefront(self, filename: str):
        with open(filename, "w") as f:
            for vertex in self.vertices:
                f.write("v " + " ".join(map(str, vertex)) + "\n")

            for norm in self.face_normals:
                f.write("vn " + " ".join(map(str, norm)) + "\n")

            for face_id, face in enumerate(self.hyperfaces):
                vn_str = "//" + str(face_id + 1)
                f.write(
                    "f " + " ".join(map(lambda x: str(x + 1) + vn_str, face)) + "\n"
                )

    def reorder_faces(self):
        if self.vertices.shape[1] != 3:
            return

        for face_id, face in enumerate(self.hyperfaces):
        # for face_id in tqdm(range(self.num_faces)):
            # check triangle direction
            face_vertices = self.vertices[face]
            face_normals = self.face_normals[face_id]
            norm = np.cross(
                face_vertices[1] - face_vertices[0],
                face_vertices[2] - face_vertices[1],
            )

            if np.dot(norm, face_normals) < 0:
                self.hyperfaces[face_id] = face[::-1]

    def get_hyperface(self, face_index: int):
        return self.vertices[self.hyperfaces[face_index]].T

    def nd_rotation(self, t: float, dim: int, ax1: int, ax2: int) -> np.ndarray:
        RM = np.eye(dim)
        cos = np.cos(t)
        sin = (-1) ** (ax1 + ax2) * np.sin(t)
        RM[ax1, ax1] = cos
        RM[ax1, ax2] = -sin
        RM[ax2, ax1] = sin
        RM[ax2, ax2] = cos

        return RM

    def project_3d_from_4d(
        self,
        rotation_list: List[Tuple[float, int, int]],
        center: np.ndarray = np.array([0.5, 0.5, 0.5, 0.5]),
    ):
        """
        rotation_list: [(angle, axis0, axis1), ...]
        """
        assert self.vertices.shape[1] == 4

        rotate_matrix = np.eye(4)
        # shift to center
        new_vertices = self.vertices - center

        for rotation in rotation_list:
            angle, axis0, axis1 = rotation
            rotate_matrix = rotate_matrix @ self.nd_rotation(angle, 4, axis0, axis1)

        new_vertices = new_vertices @ rotate_matrix.T
        new_vertices = new_vertices[:, :3]

        # shift back
        new_vertices += center[:3]

        # TODO: Update Face Normal
        hyperfaces = np.ndarray((0, 3), dtype=int)
        face_normals = np.ndarray((0, 3), dtype=float)
        patterns = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
        for face_id, face in enumerate(self.hyperfaces):
            norm_projected = self.face_normals[face_id][:-1]
            for pattern in patterns:
                new_face = face[pattern]
                face_vertices = new_vertices[new_face]
                norm = la.null_space(
                    np.vstack(
                        (
                            face_vertices[2] - face_vertices[0],
                            face_vertices[1] - face_vertices[0],
                        )
                    )
                ).T[0]
                norm *= np.sign(np.dot(norm_projected, norm))
                face_normals = np.vstack((face_normals, norm))
                hyperfaces = np.vstack((hyperfaces, new_face))

        # breakpoint()
        return Mesh(new_vertices, hyperfaces, face_normals)


def split_hyperface(
    mesh_vertices: np.ndarray,
    hyperface: np.ndarray,
    hole_cuts: Tuple[np.ndarray, np.ndarray, np.ndarray],
):
    vertices, edges, edge_norms = hole_cuts
    vertices, remap = remove_redundant_vertices(vertices)
    edges = remap_vertex_indexes(edges, remap)
    cut_edges_groups = generate_cut_groups(vertices, edges)

    ##############################################
    # Project holes to mesh
    ##############################################
    hyperface_vertices = mesh_vertices[hyperface]
    O = hyperface_vertices[0:1].T
    V = hyperface_vertices[1:].T - O
    Vinv = la.pinv(V)

    vertices = (Vinv @ (vertices.T - O)).T
    hyperface_vertices = (Vinv @ (hyperface_vertices.T - O)).T

    # NOTE: Calculate edge norm after transform
    misaligned_norms = (Vinv @ edge_norms.T).T
    new_edge_norms = []
    for edge, misaligned_norm in zip(edges, misaligned_norms):
        edge_verts = vertices[edge]
        vecs = edge_verts[1:] - edge_verts[:1]
        new_norm = la.null_space(vecs).T[0]
        direction = np.dot(new_norm, misaligned_norm)
        if direction < 0:
            new_norm *= -1.0
        new_edge_norms.append(new_norm)

    edge_norms = np.array(new_edge_norms)

    # print(vertices)
    # print(edge_norms)

    ##############################################
    # Find connectable meshes
    ##############################################
    # permute edges of hyperface
    hf_verts_neighbors = [
        [None for _ in range(hyperface_vertices.shape[0])]
        for _ in range(hyperface_vertices.shape[0])
    ]
    cut_hfe_neighbors = [
        [
            [None for _ in range(hyperface_vertices.shape[0])]
            for _ in range(hyperface_vertices.shape[0])
        ]
        for _ in range(len(cut_edges_groups))
    ]
    for bfvi in range(len(hyperface_vertices)):
        for bfvj in range(bfvi + 1, len(hyperface_vertices)):
            s = hyperface_vertices[bfvi]
            e = hyperface_vertices[bfvj]
            seg = e - s

            # NOTE: find all hole meshes that has vertices on this segment
            cut_vert_on_seg = []
            vert_checked = np.zeros(vertices.shape[0], dtype=bool)
            for cut_idx, cut_edges_ids in enumerate(cut_edges_groups):
                for cut_edge in cut_edges_ids:
                    edge = edges[cut_edge]
                    for vertex_idx in edge:
                        if vert_checked[vertex_idx]:
                            continue

                        vert_checked[vertex_idx] = True
                        vertex = vertices[vertex_idx]
                        vs = vertex - s

                        len_seg = np.linalg.norm(seg)
                        len_vs = np.linalg.norm(vs)

                        cos = np.dot(seg, vs) / (len_seg * len_vs)
                        if np.abs(1 - cos) < TOL:
                            norm_proj = np.dot(vs, edge_norms[cut_edge])
                            direction = 1 if norm_proj > 0 else -1
                            cut_vert_on_seg.append(
                                (len_vs / len_seg, direction, cut_idx, vertex_idx)
                            )

            # print(cut_vert_on_seg)

            cut_vert_on_seg = sorted(cut_vert_on_seg, key=lambda x: x[0])
            to_remove = [False for _ in range(len(cut_vert_on_seg))]
            for i, (_, dir, cut_idx, vertex_idx) in enumerate(cut_vert_on_seg[:-1]):
                _, next_dir, next_cut_idx, next_vertex_idx = cut_vert_on_seg[i + 1]
                if dir == 1 and next_dir == -1:
                    cut_hfe_neighbors[cut_idx][bfvi][bfvj] = (
                        next_cut_idx,
                        next_vertex_idx,
                    )
                    cut_hfe_neighbors[cut_idx][bfvj][bfvi] = (
                        next_cut_idx,
                        next_vertex_idx,
                    )

                    if cut_idx == next_cut_idx:
                        to_remove[i] = True
                        to_remove[i + 1] = True
                        continue

                    cut_hfe_neighbors[next_cut_idx][bfvi][bfvj] = (
                        cut_idx,
                        vertex_idx,
                    )
                    cut_hfe_neighbors[next_cut_idx][bfvj][bfvi] = (
                        cut_idx,
                        vertex_idx,
                    )

            cut_vert_on_seg = [
                cut_vert_on_seg[i]
                for i in range(len(cut_vert_on_seg))
                if not to_remove[i]
            ]
            if len(cut_vert_on_seg) == 0:
                hf_verts_neighbors[bfvi][bfvj] = (-1, -1)  # pyright: ignore
                hf_verts_neighbors[bfvj][bfvi] = (-1, -1)  # pyright: ignore
                continue

            if cut_vert_on_seg[0][1] == -1:
                hf_verts_neighbors[bfvi][bfvj] = (  # pyright: ignore
                    cut_vert_on_seg[0][2],
                    cut_vert_on_seg[0][3],
                )
            if cut_vert_on_seg[-1][1] == 1:
                hf_verts_neighbors[bfvj][bfvi] = (  # pyright: ignore
                    cut_vert_on_seg[-1][2],
                    cut_vert_on_seg[-1][3],
                )

    # for a in cut_hfe_neighbors:
    #     print(a)

    bv_isolated = [
        all([neighbor is None or neighbor[0] < 0 for neighbor in neighbors])
        for neighbors in hf_verts_neighbors
    ]
    # print(bv_isolated)

    num_valid_bv = 0
    bv_vertice_ids = [-1 for _ in range(len(hyperface_vertices))]
    for bfvi in range(len(hyperface_vertices)):
        if not bv_isolated[bfvi]:
            bv_vertice_ids[bfvi] = vertices.shape[0] + num_valid_bv
            num_valid_bv += 1
    vertices = np.vstack((vertices, hyperface_vertices))

    for i in range(len(hyperface_vertices)):
        bf_vert_ids = [j for j in range(len(hyperface_vertices)) if j != i]
        bf_cut_groups = [-1 for _ in range(len(cut_edges_groups))]
        bf_bv_groups = [-1 for _ in range(len(bf_vert_ids))]
        group_count = 0
        for bfvi_id, bfvi in enumerate(bf_vert_ids):
            if bv_isolated[bfvi]:
                continue

            if bf_bv_groups[bfvi_id] < 0:
                bf_bv_groups[bfvi_id] = group_count
                group_count += 1

            for bfvj_id, bfvj in enumerate(bf_vert_ids):
                if bfvi == bfvj:
                    continue
                neighbor = hf_verts_neighbors[bfvi][bfvj]
                if neighbor is None:
                    continue

                current_group = bf_bv_groups[bfvi_id]
                if neighbor[0] < 0:
                    bf_bv_groups[bfvj_id] = current_group
                else:
                    bf_cut_groups[neighbor[0]] = current_group

        # NOTE: cut pairs
        for cut_idx, cut_edges_ids in enumerate(cut_edges_groups):
            if all(
                [
                    all([cut_hfe_neighbors[cut_idx][i][j] is None for j in bf_vert_ids])
                    for i in bf_vert_ids
                ]
            ):
                continue

            if bf_cut_groups[cut_idx] < 0:
                bf_cut_groups[cut_idx] = group_count
                group_count += 1

            for bfvi_id, bfvi in enumerate(bf_vert_ids):
                for bfvj_id, bfvj in enumerate(bf_vert_ids):
                    if bfvi == bfvj:
                        continue

                    neighbor = cut_hfe_neighbors[cut_idx][bfvi][bfvj]
                    if neighbor is None:
                        continue

                    current_group = bf_cut_groups[cut_idx]
                    bf_cut_groups[neighbor[0]] = current_group

        # NOTE: find cut's points on this boundary face
        bf_vertices = hyperface_vertices[bf_vert_ids]
        bfO = bf_vertices[0:1].T
        bfV = bf_vertices[1:].T - bfO

        # WARNING: Only valid for 2D case now
        # To support 3D case, need to connenct bv with vertices that has only
        # one non-zero entry in the projection
        # The direction of 2D boundary of a 3D face is also needed
        cut_on_bf_groups = [[] for _ in range(len(cut_edges_groups))]
        for cut_idx, cut_edges_ids in enumerate(cut_edges_groups):
            for cut_edge in cut_edges_ids:
                edge = edges[cut_edge]
                verts = vertices[edge]
                verts_proj = (np.linalg.pinv(bfV) @ (verts.T - bfO)).T
                diff = np.linalg.norm(verts - (bfV @ verts_proj.T + bfO).T, axis=1)
                is_verts_on_face = diff < TOL

                # has_component = (verts_proj > TOL) * 1
                # on_bf_edge = np.logical_and(
                #     np.sum(has_component, axis=1) == 1, is_verts_on_face
                # )
                # print(on_bf_edge)

                # WARNING: Need to construct point order for 3D case
                if len(is_verts_on_face) > 0:
                    cut_on_bf_groups[cut_idx].extend(
                        edge[np.where(is_verts_on_face)].tolist()
                    )

        bf_norm = la.null_space(bfV.T).T
        bf_norm /= np.linalg.norm(bf_norm)

        if np.dot(bf_norm, hyperface_vertices[i] - bf_vertices[0]) < 0:
            bf_norm *= -1

        # print(bf_cut_groups)
        # print(bf_bv_groups)

        # TODO: fix holes on boundary face in 3D
        for g in range(group_count):
            new_cut_edge = []
            for cut_idx, bf_cut_group in enumerate(bf_cut_groups):
                if bf_cut_group == g:
                    new_cut_edge.extend(cut_on_bf_groups[cut_idx])
            for bv_idx, bf_bv_group in enumerate(bf_bv_groups):
                if bf_bv_group == g:
                    new_cut_edge.append(bv_vertice_ids[bf_vert_ids[bv_idx]])

            if len(new_cut_edge) > 0:
                # WARNING: Just a hot fix
                # if len(new_cut_edge) == 2:
                edges = np.vstack((edges, new_cut_edge))
                edge_norms = np.vstack((edge_norms, bf_norm))

    # TODO: Split groups again
    cut_edges_groups = generate_cut_groups(vertices, edges)
    # print(cut_edges_groups)

    new_vertices = np.empty((0, 2), dtype=float)
    new_hyperfaces = np.empty((0, 3), dtype=int)
    for cut_edges_ids in cut_edges_groups:
        cut_edges = edges[cut_edges_ids]

        vert_included = [False for _ in range(vertices.shape[0])]
        new_vert_ids = [i for i in range(vertices.shape[0])]
        old_vert_ids = []
        group_vertices = []
        for cut_edge in cut_edges:
            for vertex in cut_edge:
                if not vert_included[vertex]:
                    new_vert_ids[vertex] = len(group_vertices)
                    group_vertices.append(vertices[vertex])
                    old_vert_ids.append(vertex)

                vert_included[vertex] = True

        map_to_new = np.vectorize(lambda x: new_vert_ids[x], cache=True)

        group_vertices = np.array(group_vertices)
        group_edges = map_to_new(cut_edges)

        # WARNING: Just a hot fix
        # if group_vertices.shape[0] < 3:
        #     continue

        info = triangle.MeshInfo()
        info.set_points(group_vertices)
        info.set_facets(group_edges)

        tri = triangle.build(info, volume_constraints=False)
        group_vertices = np.array(tri.points)
        group_triangles = np.array(tri.elements)

        # WARNING: Just a hot fix
        # if group_vertices.shape[0] == 0:
        #     continue

        num_new_vertices = new_vertices.shape[0]
        group_triangles += num_new_vertices

        new_vertices = np.vstack((new_vertices, group_vertices))
        new_hyperfaces = np.vstack((new_hyperfaces, group_triangles))

    # TODO: There are some useless vertices, remove them

    # print(new_vertices)
    # print(hyperface_vertices)
    # print(hyperface)
    new_vertices, remap = remove_vertices_by_pos(
        new_vertices, hyperface_vertices, hyperface - hyperface_vertices.shape[0]
    )
    new_hyperfaces = remap_vertex_indexes(new_hyperfaces, remap)
    new_hyperfaces += mesh_vertices.shape[0]

    new_vertices = (V @ new_vertices.T + O).T

    return new_vertices, new_hyperfaces


def generate_cut_groups(vertices: np.ndarray, edges: np.ndarray):
    connected_edges = [[] for _ in range(edges.shape[0])]
    for bi in range(edges.shape[0]):
        for bj in range(edges.shape[0]):
            if bi == bj:
                continue

            # check common vertices
            edge0 = edges[bi]
            edge1 = edges[bj]
            common = np.intersect1d(edge0, edge1, assume_unique=True)
            if len(common) > 0:
                connected_edges[bi].append(bj)

    visited = np.zeros(edges.shape[0], dtype=bool)
    cut_groups = []
    for bi in range(edges.shape[0]):
        if visited[bi]:
            continue

        component = []
        stack = [bi]
        while len(stack) > 0:
            edge = stack.pop()
            if visited[edge]:
                continue

            visited[edge] = True
            component.append(edge)
            stack += connected_edges[edge]

        cut_groups.append(component)

    return cut_groups


def remove_vertices_by_pos(
    vertices: np.ndarray,
    vertices_to_remove: np.ndarray,
    new_indexes_for_removed: np.ndarray,
):
    N = vertices.shape[0]

    remap = np.zeros(N, dtype=int)
    removed = np.zeros(N, dtype=bool)
    for vertex, index in zip(vertices_to_remove, new_indexes_for_removed):
        diff = la.norm(vertices - vertex, axis=1)
        identical = diff < TOL
        removed = np.logical_or(removed, identical)
        remap[identical] = index

    remap[~removed] = np.arange(np.sum(~removed))

    return vertices[~removed], remap


# Remove redundant vertices on holes
def remove_redundant_vertices(vertices: np.ndarray):
    N = vertices.shape[0]

    # NOTE: Check duplicated vertices
    exists = np.ones(N, dtype=bool)
    remap = np.array([i for i in range(N)], dtype=int)

    for i in range(N):
        if not exists[i]:
            continue

        identicals = np.logical_and(
            exists, np.sqrt(np.sum(np.square(vertices[i] - vertices), axis=1)) < TOL
        )
        identicals[i] = False

        if len(identicals) > 0:
            exists[identicals] = False
            remap[identicals] = i

    prefix_counts = np.zeros(N, dtype=int)
    prefix_counts[0] = 1 if exists[0] else 0
    for i in range(1, N):
        prefix_counts[i] = prefix_counts[i - 1] + (1 if exists[i] else 0)

    shrink_map = np.zeros(N, dtype=int)
    shrink_map[exists] = prefix_counts[exists] - 1

    for i in range(N):
        remap[i] = shrink_map[remap[i]]

    return vertices[exists], remap


def remap_vertex_indexes(faces: np.ndarray, remap: np.ndarray):
    map_func = np.vectorize(lambda x: remap[x], cache=True)
    return map_func(faces)
