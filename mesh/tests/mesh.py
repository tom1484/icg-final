import matplotlib.pyplot as plt
import numpy as np
import tqdm
from scipy import linalg as la

from ..config import TOL

from ..convex import polyhedron_from_halfspaces
from ..mesh import Mesh
from ..plot import plot_3D_mesh
from ..space import KSimplexSpace, nd_rotation
from ..split_4D import split_4D_hyperface
from ..triangle import triangulate


def init_sculpt():
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

    hyperfaces = np.ndarray((0, 4), dtype=int)
    face_normals = []
    for i in range(4):

        base = 2**i
        mod_pattern = pattern % base
        new_pattern = (pattern - mod_pattern) * 2 + mod_pattern
        new_faces = np.vstack((new_pattern, new_pattern + base))

        for new_face in new_faces:
            face_vertices = vertices[new_face]
            norm = la.null_space(
                np.vstack(
                    (
                        face_vertices[3] - face_vertices[0],
                        face_vertices[2] - face_vertices[0],
                        face_vertices[1] - face_vertices[0],
                    )
                )
            ).T[0]

            # determine direction
            from_center = face_vertices[0] - 0.5
            norm *= np.sign(np.dot(from_center, norm))

            face_normals.append(norm)

        hyperfaces = np.vstack((hyperfaces, new_faces))

    return Mesh(vertices, hyperfaces, np.array(face_normals))


def get_sculpt_tetrahedron(sculpt, index):
    tetrahedrons_id = sculpt["tetrahedrons_id"]
    return sculpt["vertices"][tetrahedrons_id[index]].T


sculpt = init_sculpt()


def generate_cuts(mesh: Mesh, RM: np.ndarray, axis):
    hyperfaces = []
    face_normals = []
    patterns = np.array([[0, 1, 2, 3], [3, 4, 5, 1], [1, 2, 3, 5]])

    vertices = (mesh.vertices - 0.5) @ RM.T + 0.5
    edges = mesh.hyperfaces
    edge_normals = mesh.face_normals @ RM.T

    plane0 = np.hstack(
        (
            vertices[:, :axis],
            np.zeros((vertices.shape[0], 1)),
            vertices[:, axis:],
        )
    )
    plane1 = plane0.copy()

    plane0[:, axis] = -0.2
    plane1[:, axis] = 1.2

    num_verts = vertices.shape[0]
    vertices = np.vstack((plane0, plane1))

    for edge_id, vert_ids in enumerate(edges):
        vert_ids = np.hstack((vert_ids, vert_ids + num_verts))
        hyperface = vert_ids[patterns[0]]

        for pattern in patterns:
            hyperface = vert_ids[pattern]
            hyperfaces.append(hyperface)

            face_normal = edge_normals[edge_id]
            norm = np.hstack((face_normal[:axis], 0, face_normal[axis:]))
            face_normals.append(norm)

    return Mesh(vertices, np.array(hyperfaces), np.array(face_normals))


# def generate_cuts(mesh, RM):
#     cuts = []
#     patterns = np.array([[0, 1, 2, 3], [3, 4, 5, 1], [1, 2, 3, 5]])
#
#     for _, face in enumerate(mesh["faces"]):
#         vertices = mesh["vertices"][face] @ RM.T
#
#         vertices0 = np.pad(vertices, ((0, 0), (0, 1)), constant_values=-0.1)
#         vertices1 = np.pad(vertices, ((0, 0), (0, 1)), constant_values=1.1)
#
#         vertices = np.vstack((vertices0, vertices1))
#
#         for pattern in patterns:
#             tetrahedron = vertices[pattern].T
#             cuts.append(tetrahedron)
#
#     return cuts

# TODO: Revert the mesh for cutting
mesh = Mesh(
    np.array(
        [
            [0.2, 0.2, 0.2],
            [0.8, 0.2, 0.2],
            [0.2, 0.8, 0.2],
            [0.8, 0.8, 0.2],
            [0.2, 0.2, 0.8],
            [0.8, 0.2, 0.8],
            [0.2, 0.8, 0.8],
            [0.8, 0.8, 0.8],
        ],
        dtype=np.float64,
    ),
    np.array(
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
        ],
        dtype=np.int32,
    ),
    np.array(
        [
            [0, 0, -1],
            [0, 0, -1],
            [0, -1, 0],
            [0, -1, 0],
            [-1, 0, 0],
            [-1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
        ],
        dtype=np.float64,
    ),
)
# RM = nd_rotation(0.7, 3, 0, 2)
RM = np.identity(3)

cuts0 = generate_cuts(mesh, RM, 3)
cuts1 = generate_cuts(mesh, RM, 0)

# Display
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.set_xlim([-0.2, 1.2])
ax.set_ylim([-0.2, 1.2])
ax.set_zlim([-0.2, 1.2])


def plot_polygon(vertices, color="blue"):
    verts = np.vstack((vertices, vertices[0:1]))
    ax.plot(verts[:, 0], verts[:, 1], verts[:, 2], color=color)


def project_norm_to_hyperface(norm: np.ndarray, hyperface: np.ndarray):
    proj = norm - np.dot(norm, hyperface) * hyperface / np.dot(hyperface, hyperface)
    return proj / np.linalg.norm(proj)


def perform_cut(cuts, sculpt, debug=False) -> Mesh:
    # hole's (boundary points, edges, edge directions)
    sculpt_face_holes = [[[], [], []] for _ in range(sculpt.num_faces)]
    cuts_face_holes = [[[], [], []] for _ in range(cuts.num_faces)]

    for cut_idx in range(cuts.num_faces):
        T0 = cuts.get_hyperface(cut_idx)
        S0 = KSimplexSpace(T0)

        for sculpt_idx in range(sculpt.num_faces):
            T1 = sculpt.get_hyperface(sculpt_idx)
            S1 = KSimplexSpace(T1)
            Si = KSimplexSpace.space_intersect(S0, S1)

            if Si.k < Si.dim - 2:
                continue

            pA1, pb1 = S0.restrict_subspace(Si)
            pA2, pb2 = S1.restrict_subspace(Si)

            pA = np.vstack((pA1, pA2))
            pb = np.vstack((pb1, pb2))

            intersections_p, edges = polyhedron_from_halfspaces(pA, pb)

            # Non-paralell case
            try:
                if intersections_p.T.shape[0] > 1:
                    if Si.k == Si.dim - 2:
                        # Ensure that the intersection is a polygon
                        if intersections_p.shape[0] <= 2:
                            continue

                        # NOTE: Edges are exact pairs in this case
                        edges = np.array(edges)
                        _, edges = triangulate(intersections_p, np.array(edges))
                        intersections = (Si.O + Si.V @ intersections_p.T).T

                        # Add vertices to holes on faces
                        cut_norm = cuts.face_normals[cut_idx]
                        sculpt_norm = sculpt.face_normals[sculpt_idx]

                        cuts_faces_hole = cuts_face_holes[cut_idx]
                        cuts_faces_hole[1].extend(edges + len(cuts_faces_hole[0]))
                        # cuts_faces_hole[1].append(edges + len(cuts_faces_hole[0]))
                        cuts_faces_hole[0].extend(intersections)
                        cuts_faces_hole[2].extend(
                            np.repeat(
                                -project_norm_to_hyperface(
                                    sculpt_norm, cut_norm
                                ).reshape((1, 4)),
                                len(edges),
                                axis=0,
                            )
                        )
                        # cuts_faces_hole[2].append(
                        #     -project_norm_to_hyperface(sculpt_norm, cut_norm)
                        # )

                        sculpt_faces_hole = sculpt_face_holes[sculpt_idx]
                        sculpt_faces_hole[1].extend(edges + len(sculpt_faces_hole[0]))
                        sculpt_faces_hole[0].extend(intersections)
                        sculpt_faces_hole[2].extend(
                            np.repeat(
                                -project_norm_to_hyperface(
                                    cut_norm, sculpt_norm
                                ).reshape((1, 4)),
                                len(edges),
                                axis=0,
                            )
                        )

                    # Paralell case
                    else:
                        pass

            except FloatingPointError as e:
                raise e

    all_vertices = np.vstack((sculpt.vertices, cuts.vertices))
    sculpt_hyperfaces = sculpt.hyperfaces
    cuts_hyperfaces = cuts.hyperfaces + sculpt.num_verts

    new_hyperfaces = np.empty((0, 4), dtype=np.int32)
    new_normals = np.empty((0, 4), dtype=np.float64)

    for i in range(cuts.num_faces):
        # TODO: Simplify hole
        # vertices = cuts_face_holes[i][0]
        # faces = cuts_face_holes[i][1]
        # normals = cuts_face_holes[i][2]
        #
        # groups = []
        # group_normals = []
        # group_count = 0
        # face_groups = [-1 for _ in range(len(faces))]
        # for j, normal in enumerate(normals):
        #     found = False
        #     for k, group_normal in enumerate(group_normals):
        #         if np.all(np.abs(normal - group_normal) < TOL):
        #             groups[k].append(faces[j])
        #             face_groups[j] = k
        #             found = True
        #             break
        #     if not found:
        #         groups.append([faces[j]])
        #         group_normals.append(normal)
        #         face_groups[j] = group_count
        #         group_count += 1
        #
        # for group in groups:
        #     group = np.vstack(group)
        #     group = np.sort(group, axis=1)
        #     group = np.unique(group, axis=0)

        h_vertices = np.array(cuts_face_holes[i][0])
        h_faces = np.array(cuts_face_holes[i][1])
        h_normals = np.array(cuts_face_holes[i][2])

        if h_vertices.shape[0] == 0:
            continue

        hyperface = cuts_hyperfaces[i]

        new_verts, new_hfs = split_4D_hyperface(
            ax, all_vertices, hyperface, (h_vertices, h_faces, h_normals)
        )
        new_hyperfaces = np.vstack((new_hyperfaces, new_hfs))
        all_vertices = np.vstack((all_vertices, new_verts))
        new_normals = np.vstack(
            (
                new_normals,
                np.repeat(cuts.face_normals[i : i + 1], new_hfs.shape[0], axis=0),
            )
        )

    for i in range(sculpt.num_faces):
        # for i in tqdm.tqdm(range(sculpt.num_faces)):
        h_vertices = np.array(sculpt_face_holes[i][0])
        h_faces = np.array(sculpt_face_holes[i][1])
        h_normals = np.array(sculpt_face_holes[i][2])

        if h_vertices.shape[0] == 0:
            continue

        hyperface = sculpt_hyperfaces[i]

        new_verts, new_hfs = split_4D_hyperface(
            ax, all_vertices, hyperface, (h_vertices, h_faces, h_normals)
        )
        new_hyperfaces = np.vstack((new_hyperfaces, new_hfs))
        all_vertices = np.vstack((all_vertices, new_verts))
        new_normals = np.vstack(
            (
                new_normals,
                np.repeat(sculpt.face_normals[i : i + 1], new_hfs.shape[0], axis=0),
            )
        )

    # print(new_normals)

    # WARNING: Remove this
    # remove_faces = np.ones(new_hyperfaces.shape[0], dtype=bool)
    # for i, face in enumerate(new_hyperfaces):
    #     face_vertices = all_vertices[face]
    #     # print(np.abs(face_vertices - 0.5))
    #     remove_faces[i] = np.all(np.abs(face_vertices[:, :3] - 0.5) < 0.4)
    # new_hyperfaces = new_hyperfaces[~remove_faces]
    # new_normals = new_normals[~remove_faces]

    return Mesh(all_vertices, new_hyperfaces, new_normals)


sculpt0 = perform_cut(cuts0, sculpt, debug=True)
# sculpt1 = perform_cut(cuts1, sculpt0, debug=True)

sculpt_3D = sculpt0.project_3d_from_4d(
    [
        # (np.pi / 4, 0, 3),
        # (np.pi / 4, 1, 3),
        # (np.pi / 4, 2, 3),
        # (np.pi / 4, 0, 1),
        # (np.pi / 4, 0, 2),
        # (np.pi / 4, 1, 2),
    ]
)
sculpt_3D.reorder_faces()
sculpt_3D.to_wavefront("mesh/result/sculpt_4D.obj")

# print(sculpt_3D.hyperfaces)
# print(sculpt_3D.face_normals)
# plot_3D_mesh(ax, sculpt_3D, plot_normal=True)
# plt.show()
