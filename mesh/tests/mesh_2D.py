import matplotlib.pyplot as plt
import numpy as np
import tqdm
from scipy import linalg as la

from ..convex import polyhedron_from_halfspaces
from ..mesh import Mesh
from ..space import KSimplexSpace, nd_rotation
from ..split_3D import split_3D_hyperface
from ..plot import plot_3D_mesh

np.seterr(all="raise")


def init_sculpt():
    # Create initial 3D cube
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )

    pattern = np.array(
        [
            [0, 1, 3],
            [0, 2, 3],
        ],
        dtype=np.int32,
    )

    hyperfaces = np.ndarray((0, 3), dtype=int)
    face_normals = []
    for i in range(3):
        base = 2**i
        mod_pattern = pattern % base
        new_pattern = (pattern - mod_pattern) * 2 + mod_pattern
        new_faces = np.vstack((new_pattern, new_pattern + base))

        for new_face in new_faces:
            face_vertices = vertices[new_face]
            norm = la.null_space(
                np.vstack(
                    (
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


sculpt = init_sculpt()


def generate_cuts(mesh: Mesh, RM: np.ndarray, axis):
    hyperfaces = []
    face_normals = []
    patterns = np.array([[0, 1, 2], [1, 2, 3]])

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


mesh = Mesh(
    np.array(
        [
            # [0.2, 0.2],
            # [0.8, 0.2],
            # [0.2, 0.8],
            # [0.8, 0.8],
            [0.2, 0.2],
            [0.7, 0.2],
            [0.2, 0.7],
            [0.8, 0.8],
            [0.3, 0.8],
            [0.8, 0.3],
        ],
        dtype=np.float64,
    ),
    np.array(
        [
            # [0, 1],
            # [0, 2],
            # [1, 3],
            # [2, 3],
            [0, 1],
            [1, 2],
            [2, 0],
            [3, 4],
            [4, 5],
            [5, 3],
        ],
        dtype=np.int32,
    ),
    np.array(
        [
            [0, -1],
            [1, 1],
            [-1, 0],
            [0, 1],
            [-1, -1],
            [1, 0],
        ],
        dtype=np.float64,
    ),
)
RM = nd_rotation(0.6, 2, 0, 1)
# RM = np.identity(2)

cuts0 = generate_cuts(mesh, RM, 0)
cuts1 = generate_cuts(mesh, RM, 1)


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


def perform_cut(cuts, sculpt, debug=False):
    # hole's (boundary points, edges, edge directions)
    sculpt_face_holes = [[np.empty((0, 3)), [], []] for _ in range(sculpt.num_faces)]
    cuts_face_holes = [[np.empty((0, 3)), [], []] for _ in range(cuts.num_faces)]

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

            intersections_p, _ = polyhedron_from_halfspaces(pA, pb)

            # Non-paralell case
            if intersections_p.T.shape[0] > 1:
                if Si.k == Si.dim - 2:
                    intersections = (Si.O + Si.V @ intersections_p.T).T
                    # ax.scatter(intersections[:, 0], intersections[:, 1], intersections[:, 2])
                    # Add vertices to holes on faces

                    cut_norm = cuts.face_normals[cut_idx]
                    sculpt_norm = sculpt.face_normals[sculpt_idx]

                    cuts_faces_hole = cuts_face_holes[cut_idx]
                    cuts_faces_hole[0] = np.vstack(
                        (cuts_faces_hole[0], intersections)
                    )
                    edge = list(
                        range(
                            cuts_faces_hole[0].shape[0] - intersections.shape[0],
                            cuts_faces_hole[0].shape[0],
                        )
                    )
                    cuts_faces_hole[1].append(edge)
                    cuts_faces_hole[2].append(
                        -project_norm_to_hyperface(sculpt_norm, cut_norm)
                    )

                    sculpt_faces_hole = sculpt_face_holes[sculpt_idx]
                    sculpt_faces_hole[0] = np.vstack(
                        (sculpt_faces_hole[0], intersections)
                    )
                    edge = list(
                        range(
                            sculpt_faces_hole[0].shape[0] - intersections.shape[0],
                            sculpt_faces_hole[0].shape[0],
                        )
                    )
                    sculpt_faces_hole[1].append(edge)
                    sculpt_faces_hole[2].append(
                        -project_norm_to_hyperface(cut_norm, sculpt_norm)
                    )

                # Paralell case
                else:
                    pass

    all_vertices = np.vstack((sculpt.vertices, cuts.vertices))
    sculpt_hyperfaces = sculpt.hyperfaces
    cuts_hyperfaces = cuts.hyperfaces + sculpt.num_verts

    new_vertices = np.empty((0, 3), dtype=np.float64)
    new_hyperfaces = np.empty((0, 3), dtype=np.int32)
    new_normals = np.empty((0, 3), dtype=np.float64)

    for i in range(cuts.num_faces):
        # for i in tqdm.tqdm(range(cuts.num_faces)):
        h_vertices = cuts_face_holes[i][0]
        h_faces = np.array(cuts_face_holes[i][1])
        h_normals = np.array(cuts_face_holes[i][2])

        if h_vertices.shape[0] == 0:
            continue

        hyperface = cuts_hyperfaces[i]

        new_verts, new_hfs = split_3D_hyperface(
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
        h_vertices = sculpt_face_holes[i][0]
        h_faces = np.array(sculpt_face_holes[i][1])
        h_normals = np.array(sculpt_face_holes[i][2])

        if h_vertices.shape[0] == 0:
            continue

        hyperface = sculpt_hyperfaces[i]

        new_verts, new_hfs = split_3D_hyperface(
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

    return Mesh(all_vertices, new_hyperfaces, new_normals)


# sculpt0 = perform_cut(cuts0, sculpt)
# sculpt1 = perform_cut(cuts1, sculpt0, debug=True)
sculpt1 = perform_cut(cuts1, cuts0)
print(sculpt.hyperfaces)

# print(sculpt1.num_verts)
# print(sculpt1.num_faces)

# plot_3D_mesh(sculpt, color="blue")
# plot_3D_mesh(cuts0, color="green")

# plot_3D_mesh(sculpt0, color="red")
# plot_3D_mesh(cuts1, color="green")
# sculpt0.reorder_faces()
# sculpt0.to_wavefront("mesh/result/sculpt0.obj")

# def plot_mesh_gif(mesh: Mesh):
#     fig = plt.figure()
#     ax = plt.axes(projection="3d")
#     ax.set_xlim([-0.2, 1.2])
#     ax.set_ylim([-0.2, 1.2])
#     ax.set_zlim([-0.2, 1.2])
#
#     image_array = []
#     image = np.empty((600, 800, 4), dtype=np.uint8)
#
#     for i in range(mesh.num_faces):
#         plt.cla()
#         verts = mesh.get_hyperface(i)
#         verts = np.vstack((verts, verts[0:1]))
#         ax.plot(verts[:, 0], verts[:, 1], verts[:, 2], color="blue")
#         # plt.draw()
#         plt.imshow(image)

plot_3D_mesh(ax, sculpt1, color="red")
# plot_3D_mesh(cuts1, color="green")
# plot_mesh_gif(sculpt1)
sculpt1.reorder_faces()
sculpt1.to_wavefront("mesh/result/sculpt1.obj")

# for edge in new_hyperfaces:
#     h_verts = all_vertices[edge]
#     h_verts = np.vstack((h_verts, h_verts[0:1]))
#     ax.plot(h_verts[:, 0], h_verts[:, 1], h_verts[:, 2], color="red")

plt.show()
