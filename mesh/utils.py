import numpy as np
from scipy import linalg as la

from .config import TOL


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


def remove_unused_vertices(vertices: np.ndarray, faces: np.ndarray):
    N = vertices.shape[0]

    exists = np.zeros(N, dtype=bool)
    for face in faces:
        exists[face] = True

    remap = np.zeros(N, dtype=int)
    remap[exists] = np.arange(np.sum(exists))

    return vertices[exists], remap


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


def extract_vertices_from_edges(vertices: np.ndarray, edges: np.ndarray):
    used_vertex_ids = np.unique(edges.flatten())
    num_new_vertices = len(used_vertex_ids)

    new_vert_ids = np.array([i for i in range(vertices.shape[0])])
    old_vert_ids = np.array([i for i in range(num_new_vertices)])

    new_vert_ids[used_vertex_ids] = old_vert_ids
    old_vert_ids = used_vertex_ids
    # print(new_vert_ids)
    # print(old_vert_ids)

    return (
        vertices[used_vertex_ids],
        np.vectorize(lambda x: new_vert_ids[x]),
        np.vectorize(lambda x: old_vert_ids[x]),
    )


def reorder_triangles_by_norm(vertices: np.ndarray, faces: np.ndarray, face_normals: np.ndarray):
    for f, face in enumerate(faces):
        verts = vertices[face]
        edge_dir = np.cross(
            verts[1] - verts[0], verts[2] - verts[0]
        )
        if np.dot(edge_dir, face_normals[f]) < 0:
            faces[f] = np.flip(face)

    return faces
