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
        border = 0
        border2 = border + 2
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
        radius2 = radius - 2
        shape = (X**2 + Y**2 + Z**2) < (radius**2)
        if skeleton:
            shape2 = (X**2 + Y**2 + Z**2) > (radius2**2)
            shape = shape * shape2
            shape = shape.astype(np.int64) * 128
            inner_shape = 1 - shape2
            inner_shape = inner_shape.astype(np.int64) * 128
            shape = shape + inner_shape
        else:
            shape = shape.astype(np.int64) * 255
    elif type == 2:
        res = np.load(f"shapes/3d/model2_{imgSize}.npy")
    elif type == 3:
        # tetrahedron
        # shape = np.zeros((size, size, size))
        # draw the inner
        # (0, 0, 0), (0, size, 0), (size, size//2, 0), (size * 2//3, size//2, size)
        X, Y, Z = np.meshgrid(np.arange(size), np.arange(size), np.arange(size))
        shape = (
            (X - Z * 2 / 3 >= 0)
            & (X / 2 - Y + Z / 6 <= 0)
            & (X / 2 + Y + Z / 6 <= size)
        )
        if skeleton:
            shape = shape.astype(np.int64) * 128
            # edge
            # (0, 0, 0), (0, size, 0)
            shape[0:size, 0:2, 0:2] = 255
            # (0, 0, 0), (size, size//2, 0)
            shape1 = (X - 2 * Y >= 0) & (X - 2 * Y < 2) & (Z < 2)
            shape1 = shape1.astype(np.int64) * 255
            shape = np.maximum(shape, shape1)

            # (0, 0, 0), (size * 2//3, size//2, size)
            shape2 = (
                (2 * Y - Z >= -4)
                & (2 * Y - Z <= 0)
                & (X * 3 / 2 - Z >= -2)
                & (X * 3 / 2 - Z < 0)
            )
            shape2 = shape2.astype(np.int64) * 255
            shape = np.maximum(shape, shape2)

            # (0, size, 0), (size, size//2, 0)
            shape3 = (X + 2 * Y <= 2 * size) & (X + 2 * Y >= 2 * size - 4) & (Z < 2)
            shape3 = shape3.astype(np.int64) * 255
            shape = np.maximum(shape, shape3)

            # (0, size, 0), (size * 2//3, size//2, size)
            shape4 = (
                (2 * Y + Z >= 2 * size - 4)
                & (2 * Y + Z <= 2 * size)
                & (X * 3 / 2 - Z >= 0)
                & (X * 3 / 2 - Z < 2)
            )
            shape4 = shape4.astype(np.int64) * 255
            shape = np.maximum(shape, shape4)

            # (size, size//2, 0), (size * 2//3, size//2, size)
            shape5 = (
                (3 * X + Z <= 3 * size)
                & (3 * X + Z >= 3 * size - 6)
                & (Y >= size / 2 - 1)
                & (Y < size / 2 + 1)
            )
            shape5 = shape5.astype(np.int64) * 255
            shape = np.maximum(shape, shape5)
        else:
            shape = shape.astype(np.int64) * 255

    elif type == 4:
        # triangular prism
        shape = np.zeros((size, size, size))
    else:
        raise ValueError("Invalid type")

    if type != 2:
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


def crop3dfrom5d(cube, shape, apply_dim):
    shape4d = np.expand_dims(shape, axis=apply_dim)
    cube = np.minimum(cube, shape4d)
    return cube


def getMesh3dfrom5d(mesh4d, turning_list):
    for turning in turning_list:
        axis1, axis2, angle = turning
        if angle != 0:
            # mesh4d = scipy.ndimage.rotate(mesh4d, angle, axes=(axis1, axis2), reshape=False)
            mesh4d = myRotate(mesh4d, angle, axes=(axis1, axis2))
    mesh3d = np.max(mesh4d, axis=(0, 1))
    return mesh3d


def getSettingList3D():
    setting_list = []
    for ang in range(0, 90, 10):
        setting_list.append((ang, 0, 0, 0, 0, 0))
    setting_list.append((90, 0, 0, 0, 0, 0))
    for ang in range(90, 180, 10):
        setting_list.append((ang, 0, 0, 0, 0, 0))
    return setting_list


def record3DVideo(mesh5d, setting_list, filename, yaw=0, pitch=0, roll=0, fps=10):
    frames = []
    for setting in tqdm.tqdm(setting_list):
        mesh3d = getMesh3dfrom5d(mesh5d, setting)
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

    return frames


def concatFrames2Gif(*frames_list, path):
    if len(frames_list) == 0:
        return
    final_frames = []
    final_frames.extend(frames_list[0])
    for frames in frames_list[1:]:
        final_frames.append(frames[0])
        final_frames.append(frames[0])
        final_frames.extend(frames)
    final_frames[0].save(
        path,
        save_all=True,
        append_images=final_frames,
        duration=1000 / 10,
        loop=0,
    )


def getTransferingSettingList3D(inverse=False):
    setting_list = []
    setting_list.append([(0, 2, 0), (1, 3, 0)])
    for ang in range(0, 90, 10):
        setting_list.append([(0, 2, ang), (1, 3, ang)])
    setting_list.append([(0, 2, 90), (1, 3, 90)])
    return setting_list


def getSettingList2D45():
    setting_list = []
    for angle in range(0, 180, 10):
        setting_list.append((angle + 45, angle + 45, angle))
    return setting_list

def myRotate(model, angle, axes=(0,1)):
    angle = np.radians(angle)
    matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    X, Y = np.meshgrid(np.arange(model.shape[axes[1]]), np.arange(model.shape[axes[0]]))
    X = X - model.shape[axes[1]] // 2
    Y = Y - model.shape[axes[0]] // 2

    target_axis = np.stack([X, Y], axis=-1)
    inverse_axis = np.linalg.inv(matrix)
    index_axis = target_axis @ inverse_axis
    index_X, index_Y = index_axis[:, :, 0], index_axis[:, :, 1]
    index_X = index_X + model.shape[axes[1]] // 2
    index_Y = index_Y + model.shape[axes[0]] // 2
    index_X = np.clip(index_X, 0, model.shape[axes[1]] - 1)
    index_Y = np.clip(index_Y, 0, model.shape[axes[0]] - 1)
    index_X = index_X.astype(np.int64)
    index_Y = index_Y.astype(np.int64)

    shift_model = np.moveaxis(model, axes, (0, 1))
    result = shift_model[index_Y, index_X]
    result = np.moveaxis(result, (0, 1), axes)
    return result

    


if __name__ == "__main__":
    size = 40
    imgSize = 80
    postfix = "23"
    b_r = int((imgSize - size) / 2)
    shape0 = getShape3d(0, size, imgSize, True, False)
    shape1 = getShape3d(1, size, imgSize, True, False)
    shape2 = getShape3d(2, size, imgSize, True, False)
    shape3 = getShape3d(3, size, imgSize, True, False)
    shape2 = np.transpose(shape2, (1, 2, 0))
    # shape3 = np.transpose(shape3, (1, 2, 0))

    mesh5d = np.zeros((imgSize, imgSize, imgSize, imgSize, imgSize))
    mesh5d[b_r : b_r + size, b_r : b_r + size, b_r : b_r + size, b_r : b_r + size, b_r : b_r + size] = 255
    mesh5d = crop3dfrom5d(mesh5d, shape2, (0, 1))
    mesh5d = crop3dfrom5d(mesh5d, shape3, (2, 3))
    # mesh4d = crop3dfrom4d(mesh4d, shape2, 2)
    print("Start recording 3D video")

    # mesh3d1 = getMesh3dfrom5d(mesh5d, [(0, 2, 0), (1, 3, 0)])
    # frames1 = record2DVideo(mesh3d1, getSettingList2D(), f"./output/3d_5d/mesh3d1.gif")
    # mesh3d2 = getMesh3dfrom5d(mesh5d, [(0, 2, 90), (1, 3, 90)])
    # frames2 = record2DVideo(mesh3d2, getSettingList2D(), f"./output/3d_5d/mesh3d2.gif")
    # mesh3d3 = getMesh3dfrom5d(mesh5d, [(0, 2, 45), (1, 3, 45)])
    # record2DVideo(mesh3d3, getSettingList2D(), f"./output/3d_5d/mesh3d3.gif")
    mesh3d1 = getMesh3dfrom5d(mesh5d, [(0, 2, 0), (1, 3, 0)])
    frames1 = record2DVideo(mesh3d1, getSettingList2D45(), f"./output/3d_5d/mesh3d1_{postfix}.gif")
    mesh3d2 = getMesh3dfrom5d(mesh5d, [(0, 2, 90), (1, 3, 90)])
    frames2 = record2DVideo(mesh3d2, getSettingList2D45(), f"./output/3d_5d/mesh3d2_{postfix}.gif")
    mesh3d3 = getMesh3dfrom5d(mesh5d, [(0, 2, 45), (1, 3, 45)])
    record2DVideo(mesh3d3, getSettingList2D(), f"./output/3d_5d/mesh3d3_{postfix}.gif")

    frames12 = record3DVideo(
        mesh5d,
        getTransferingSettingList3D(),
        f"./output/3d_5d/3d_shape{postfix}.gif",
        yaw=45,
        pitch=45,
        roll=0,
    )

    frames21 = frames12[::-1]

    concatFrames2Gif(
        frames1, frames12, frames2, frames21, path=f"./output/3d_5d/transferring_{postfix}.gif"
    )
