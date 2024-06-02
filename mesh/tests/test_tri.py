# def test_triangle_refine():
import math
import numpy as np
import meshpy.triangle as triangle

from matplotlib import pyplot as plt

segments = 50

points = [(1, 0), (1, 1), (-1, 1), (-1, -1), (1, -1)]
n_outer_points = len(points)

for i in range(0, segments):
    angle = i * 2 * math.pi / segments
    points.append((0.5 * math.cos(angle), 0.5 * math.sin(angle)))


def round_trip_connect(start, end):
    result = []
    for i in range(start, end):
        result.append((i, i + 1))
    result.append((end, start))
    return result


def needs_refinement(vertices, area):
    vert_origin, vert_destination, vert_apex = vertices
    bary_x = (vert_origin.x + vert_destination.x + vert_apex.x) / 3
    bary_y = (vert_origin.y + vert_destination.y + vert_apex.y) / 3

    dist_center = math.sqrt(bary_x**2 + bary_y**2)
    max_area = 100 * (math.fabs(0.002 * (dist_center - 0.3)) + 0.0001)
    return area > max_area


info = triangle.MeshInfo()
points = np.array(points)
print(points.shape)
info.set_points(points)
# info.set_holes([(0, 0)])

facets = round_trip_connect(0, n_outer_points - 1) + round_trip_connect(
    n_outer_points, len(points) - 1
)
facets = np.array(facets)
print(facets.shape)
info.set_facets(facets)

# mesh = triangle.build(info, refinement_func=needs_refinement)
mesh = triangle.build(info)
facets = np.array(list(mesh.facets))
points = np.array(list(mesh.points))

print(facets.shape)
print(points.shape)

plt.plot(points[:, 0], points[:, 1], "o")

for facet in facets:
    point1 = points[facet[0]]
    point2 = points[facet[1]]
    plt.plot([point1[0], point2[0]], [point1[1], point2[1]], "b-")

plt.show()

# triangle.write_gnuplot_mesh("triangles-unrefined.dat", mesh)
#
# mesh.element_volumes.setup()
#
# for i in range(len(mesh.elements)):
#     mesh.element_volumes[i] = -1
# for i in range(0, len(mesh.elements), 10):
#     mesh.element_volumes[i] = 1e-8
#
# mesh = triangle.refine(mesh)


def test_point_attributes():
    import meshpy.triangle as triangle

    points = [(1, 1), (-1, 1), (-1, -1), (1, -1)]
    info = triangle.MeshInfo()
    info.set_points(points)

    info.number_of_point_attributes = 2

    info.point_attributes.setup()

    for i in range(len(points)):
        info.point_attributes[i] = [0, 0]

    triangle.build(info)


def test_tetgen():
    from meshpy.tet import MeshInfo, build

    mesh_info = MeshInfo()

    mesh_info.set_points(
        [
            (0, 0, 0),
            (2, 0, 0),
            (2, 2, 0),
            (0, 2, 0),
            (0, 0, 12),
            (2, 0, 12),
            (2, 2, 12),
            (0, 2, 12),
        ]
    )

    mesh_info.set_facets(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [0, 4, 5, 1],
            [1, 5, 6, 2],
            [2, 6, 7, 3],
            [3, 7, 4, 0],
        ]
    )

    build(mesh_info)


def test_torus():
    from math import cos, pi, sin

    from meshpy.geometry import (
        EXT_CLOSED_IN_RZ,
        GeometryBuilder,
        generate_surface_of_revolution,
    )
    from meshpy.tet import MeshInfo, build

    big_r = 3
    little_r = 2.9

    points = 50
    dphi = 2 * pi / points

    rz = [
        (big_r + little_r * cos(i * dphi), little_r * sin(i * dphi))
        for i in range(points)
    ]

    geob = GeometryBuilder()
    geob.add_geometry(
        *generate_surface_of_revolution(rz, closure=EXT_CLOSED_IN_RZ, radial_subdiv=20)
    )

    mesh_info = MeshInfo()
    geob.set(mesh_info)

    build(mesh_info)


def test_tetgen_points():
    import numpy as np
    from meshpy.tet import MeshInfo, Options, build

    points = np.random.randn(10000, 3)

    mesh_info = MeshInfo()
    mesh_info.set_points(points)
    options = Options("")
    mesh = build(mesh_info, options=options)

    print(len(mesh.points))
    print(len(mesh.elements))

    # mesh.write_vtk("test.vtk")


# test_triangle_refine()
