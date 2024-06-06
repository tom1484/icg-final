import pickle

import numpy as np
import pyrender
import trimesh
from matplotlib import pyplot as plt
from tqdm import tqdm
from PIL import Image

from ..mesh import Mesh
from ..space import nd_rotation

sculpt: Mesh = pickle.load(open("mesh/result/sculpt_4D.pkl", "rb"))


# render scene
r = pyrender.OffscreenRenderer(512, 512)

# compose scene
scene = pyrender.Scene(ambient_light=[0.1, 0.1, 0.1], bg_color=[0, 0, 0])
# scene = pyrender.Scene(ambient_light=[0.1, 0.1, 0.3], bg_color=[0, 0, 0])
camera = pyrender.OrthographicCamera(xmag=1, ymag=1)
light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=2e3)
# scene.remove_node(mesh_node)
scene.add(light, pose=[[1, 0, 0, 0], [0, 0, 1, 0.5], [0, -1, 0, 0], [0, 0, 0, 1]])
scene.add(camera, pose=[[1, 0, 0, 0], [0, 0, 1, 0.5], [0, -1, 0, 0], [0, 0, 0, 1]])


renders = []


def render(PRM, RM):
    sculpt_3D = sculpt.project_3d_from_4d(PRM)
    sculpt_3D.vertices = sculpt_3D.vertices - 0.5
    sculpt_3D.vertices = sculpt_3D.vertices @ RM.T
    sculpt_3D.face_normals = sculpt_3D.face_normals @ RM.T

    tri = trimesh.Trimesh(vertices=sculpt_3D.vertices, faces=sculpt_3D.hyperfaces)
    mesh = pyrender.Mesh.from_trimesh(tri)

    mesh_node = scene.add(mesh, pose=np.eye(4))
    img, _ = r.render(scene)  # pyright: ignore
    renders.append(Image.fromarray(img))

    scene.remove_node(mesh_node)


PRM = [(0, 2, 3)]
RM = np.identity(3)

inner_rotate = 2.7

for t in tqdm(np.linspace(0, inner_rotate, 20)):
    RM = nd_rotation(t / 2, 3, 0, 2) @ nd_rotation(t, 3, 1, 2)
    render(PRM, RM)

# RM = np.identity(3)
for t in tqdm(np.linspace(0, np.pi / 2, 20)):
    PRM = [(t, 2, 3)]
    render(PRM, RM)

for t in tqdm(np.linspace(inner_rotate, 0, 20)):
    RM = nd_rotation(t / 2, 3, 0, 2) @ nd_rotation(t, 3, 1, 2)
    render(PRM, RM)


renders[0].save(
    "mesh/result/sculpt_4D.gif",
    save_all=True,
    append_images=renders[1:],
    loop=0,
    duration=1000 / 10,
)
