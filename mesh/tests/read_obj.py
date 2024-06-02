from ..mesh import Mesh


mesh = Mesh.from_wavefront("model/sofa.obj")
print(mesh.vertices)
print(mesh.hyperfaces)
