import numpy as np
import cv2
import scipy.ndimage
from PIL import Image
import tqdm
from morph2d import record2DVideo, getSettingList2D, getView2dfrom3d


def getShape3d(type, size=100, imgSize=200, skeleton=False, plot=False):
    if type == 0:
        # rectangle
        shape = np.zeros((size, size, size))
        border = int(size * 0.2)
        border2 = int(size * 0.23)
        shape[
            border : size - border, border : size - border, border : size - border
        ] = 255
        if skeleton:
            shape[
                border2 : size - border2,
                border2 : size - border2,
                border : size - border,
            ] = 128
            shape[
                border2 : size - border2,
                border : size - border,
                border2 : size - border2,
            ] = 128
            shape[
                border : size - border,
                border2 : size - border2,
                border2 : size - border2,
            ] = 128
    elif type == 1:
        x = np.asarray(range(size))
        y = np.asarray(range(size))
        z = np.asarray(range(size))
        X, Y, Z = np.meshgrid(x, y, z)
        X = X - size / 2
        Y = Y - size / 2
        Z = Z - size / 2
        radius = size * 0.3
        radius2 = size * 0.27
        shape = (X**2 + Y**2 + Z**2) < (radius**2)
        if skeleton:
            shape2 = (X**2 + Y**2 + Z**2) > (radius2**2)
            shape = shape * shape2
            shape = shape.astype(np.int64) * 255
            inner_shape = 1 - shape2
            inner_shape = inner_shape.astype(np.int64) * 128
            shape = shape + inner_shape
        else:
            shape = shape.astype(np.int64) * 255
    elif type == 2:
        res = np.load(f"shapes/3d/model2_{imgSize}.npy")
    else:
        raise ValueError("Invalid type")

    if type < 2:
        res = np.zeros((imgSize, imgSize, imgSize))
        b_r = int((imgSize - size) / 2)
        res[b_r : b_r + size, b_r : b_r + size, b_r : b_r + size] = shape
    if plot:
        setting_list = getSettingList2D()
        if skeleton:
            record2DVideo(res, setting_list, f"./shapes/3d/shape{type}_3d_skeleton.gif")
        else:
            record2DVideo(res, setting_list, f"./shapes/3d/shape{type}_3d.gif")

    return res


def crop3dfrom4d(cube, shape, apply_dim):
    shape4d = np.expand_dims(shape, axis=apply_dim)
    cube = np.minimum(cube, shape4d)
    return cube


def getMesh3dfrom4d(mesh4d, ang1, ang2, ang3, ang4, ang5, ang6):
    if ang1 != 0:
        mesh4d = scipy.ndimage.rotate(mesh4d, ang1, axes=(0, 1), reshape=False)
    if ang2 != 0:
        mesh4d = scipy.ndimage.rotate(mesh4d, ang2, axes=(0, 2), reshape=False)
    if ang3 != 0:
        mesh4d = scipy.ndimage.rotate(mesh4d, ang3, axes=(0, 3), reshape=False)
    if ang4 != 0:
        mesh4d = scipy.ndimage.rotate(mesh4d, ang4, axes=(1, 2), reshape=False)
    if ang5 != 0:
        mesh4d = scipy.ndimage.rotate(mesh4d, ang5, axes=(1, 3), reshape=False)
    if ang6 != 0:
        mesh4d = scipy.ndimage.rotate(mesh4d, ang6, axes=(2, 3), reshape=False)
    mesh3d = np.max(mesh4d, axis=0)
    return mesh3d


def getSettingList3D():
    setting_list = []
    for ang in range(0, 90, 10):
        setting_list.append((ang, 0, 0, 0, 0, 0))
    setting_list.append((90, 0, 0, 0, 0, 0))
    for ang in range(90, 180, 10):
        setting_list.append((ang, 0, 0, 0, 0, 0))
    return setting_list


def record3DVideo(mesh4d, setting_list, filename, yaw=0, pitch=0, roll=0, fps=10):
    frames = []
    for setting in tqdm.tqdm(setting_list):
        mesh3d = getMesh3dfrom4d(mesh4d, *setting)
        view = getView2dfrom3d(mesh3d, yaw, pitch, roll)
        frame = Image.fromarray(view)
        frames.append(frame)

    frames[0].save(
        filename,
        save_all=True,
        append_images=frames,
        duration=1000 / fps,
        loop=0,
    )


if __name__ == "__main__":
    size = 80
    imgSize = 100
    b_r = int((imgSize - size) / 2)
    shape0 = getShape3d(0, size, imgSize, True, False)
    shape2 = getShape3d(2, size, imgSize, True, False)

    mesh4d = np.zeros((imgSize, imgSize, imgSize, imgSize))
    mesh4d[b_r : b_r + size, b_r : b_r + size, b_r : b_r + size, b_r : b_r + size] = 255
    mesh4d = crop3dfrom4d(mesh4d, shape0, 0)
    mesh4d = crop3dfrom4d(mesh4d, shape2, 1)

    # mesh3d1 = getMesh3dfrom4d(mesh4d, 0, 0, 0, 0, 0, 0)
    # record2DVideo(mesh3d1, getSettingList2D(), f"./output/3d/mesh3d1.gif")
    # mesh3d2 = getMesh3dfrom4d(mesh4d, 90, 0, 0, 0, 0, 0)
    # record2DVideo(mesh3d2, getSettingList2D(), f"./output/3d/mesh3d2.gif")
    # mesh3d3 = getMesh3dfrom4d(mesh4d, 45, 0, 0, 0, 0, 0)
    # record2DVideo(mesh3d3, getSettingList2D(), f"./output/3d/mesh3d3.gif")

    record3DVideo(
        mesh4d, getSettingList3D(), f"./output/3d/3d_45.gif", yaw=45, pitch=45, roll=0
    )
