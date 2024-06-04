from meshpy import triangle, tet
import numpy as np


def triangulate(vertices: np.ndarray, edges: np.ndarray):
    info = triangle.MeshInfo()
    info.set_points(vertices)
    info.set_facets(edges)

    tri = triangle.build(
        info,
        volume_constraints=False,
        allow_volume_steiner=False,
        allow_boundary_steiner=False,
        quality_meshing=False,
        # generate_edges=False,
        # generate_faces=False,
        # generate_neighbor_lists=False
    )

    vertices = np.array(tri.points)
    faces = np.array(tri.elements)
    # print(np.array(tri.points).shape)
    # faces = np.sort(faces, axis=-1)
    # faces = np.unique(faces, axis=0)

    return vertices, faces


def tetrahedralize(vertices: np.ndarray, faces: np.ndarray):
    info = tet.MeshInfo()
    info.set_points(vertices)
    info.set_facets(faces)

    options = tet.Options("", quality=False)
    tetras = tet.build(info, options=options)
    # tetras = tet.build(info)
    # print(len(tetras.points))
    return np.array(tetras.elements)
