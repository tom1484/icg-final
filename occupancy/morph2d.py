import numpy as np
import cv2
from PIL import Image
import tqdm
import argparse

DEBUG = True


def getShape2d(type, size=100, imgSize=200, skeleton=False):
    if type == 0:
        # rectangle
        shape = np.zeros((size, size))
        border = 0
        border2 = border + 2
        shape[border : size - border, border : size - border] = 255
        if skeleton:
            shape[border2 : size - border2, border2 : size - border2] = 128

    elif type == 1:
        x = np.asarray(range(size))
        y = np.asarray(range(size))
        X, Y = np.meshgrid(x, y)
        X = X - size / 2
        Y = Y - size / 2
        radius = size * 0.5
        radius2 = radius - 2
        shape = (X**2 + Y**2) < (radius**2)
        if skeleton:
            shape2 = (X**2 + Y**2) > (radius2**2)
            shape = shape * shape2
            shape = shape.astype(np.int64) * 255
            inner_shape = 1 - shape2
            inner_shape = inner_shape.astype(np.int64) * 128
            shape = shape + inner_shape
        else:
            shape = shape.astype(np.int64) * 255
    else:
        raise ValueError("Invalid type")
    res = np.zeros((imgSize, imgSize))
    b_r = int((imgSize - size) / 2)
    res[b_r : b_r + size, b_r : b_r + size] = shape
    if DEBUG:
        cv2.imwrite(f"./shapes/2d/shape{type}_2d.png", res)

    return res


def crop2dfrom3d(cube, shape, apply_dim):
    shape3d = np.expand_dims(shape, axis=apply_dim)
    cube = np.minimum(cube, shape3d)
    return cube


def getView2dfrom3d(cube, yaw, pitch, roll):
    # # rotate cube
    # cube = scipy.ndimage.rotate(cube, yaw, axes=(1, 2), reshape=False)
    # cube = scipy.ndimage.rotate(cube, pitch, axes=(0, 2), reshape=False)
    # cube = scipy.ndimage.rotate(cube, roll, axes=(0, 1), reshape=False)
    cube = myRotate(cube, yaw, axes=(1, 2))
    cube = myRotate(cube, pitch, axes=(0, 2))
    cube = myRotate(cube, roll, axes=(0, 1))
    # get view
    view = np.max(cube, axis=0)
    return view


def getSettingList2D():
    setting_list = []
    for angle in range(0, 90, 10):
        setting_list.append((angle, angle, angle))
    setting_list.append((90, 90, 90))
    for angle in range(90, 180, 10):
        setting_list.append((angle, angle, angle))
    return setting_list


def record2DVideo(cube, setting_list, path_name, fps=10):
    frames = []
    for setting in tqdm.tqdm(setting_list):
        yaw, pitch, roll = setting
        view = getView2dfrom3d(cube, yaw, pitch, roll)
        image = Image.fromarray(view)
        frames.append(image)

    frames[0].save(
        path_name, save_all=True, append_images=frames, loop=0, duration=1000 / fps
    )

    return frames


def myRotate(model, angle, axes=(0,1), divide=1):
    angle = np.radians(angle)
    matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    result = np.zeros(model.shape, dtype=model.dtype)

    x_splits = np.split(np.arange(model.shape[axes[1]]), divide)
    y_splits = np.split(np.arange(model.shape[axes[0]]), divide)
    for x_range in x_splits:
        for y_range in y_splits:
            X, Y = np.meshgrid(x_range, y_range)
            center_X = X - model.shape[axes[1]] // 2
            center_Y = Y - model.shape[axes[0]] // 2

            target_axis = np.stack([center_X, center_Y], axis=-1)
            inverse_axis = np.linalg.inv(matrix)
            index_axis = target_axis @ inverse_axis
            index_X, index_Y = index_axis[:, :, 0], index_axis[:, :, 1]
            index_X = index_X + model.shape[axes[1]] // 2
            index_Y = index_Y + model.shape[axes[0]] // 2
            index_X = np.clip(index_X, 0, model.shape[axes[1]] - 1)
            index_Y = np.clip(index_Y, 0, model.shape[axes[0]] - 1)
            index_X = index_X.astype(np.int64)
            index_Y = index_Y.astype(np.int64)

            # shift_model = np.moveaxis(model, axes, (0, 1))
            if axes == (0, 1):
                result[Y, X, :] = model[index_Y, index_X, :]
            elif axes == (1, 2):
                result[:, Y, X] = model[:, index_Y, index_X]
            elif axes == (0, 2):
                result[Y, :, X] = model[index_Y, :, index_X]
            else:
                raise ValueError("Invalid axes")
    return result

if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--imgSize", type=int, default=160)

    args = argparser.parse_args()

    imgSize = args.imgSize
    size = imgSize // 2
    # 50, 80 | 80, 100
    b_r = int((imgSize - size) / 2)
    cube = np.zeros((imgSize, imgSize, imgSize))
    # cube = np.ones((size, size, size)) * 255
    cube[b_r : b_r + size, b_r : b_r + size, b_r : b_r + size] = 255
    shape1 = getShape2d(0, size, imgSize, True)
    shape2 = getShape2d(1, size + 6, imgSize, True)
    cube = crop2dfrom3d(cube, shape1, 0)
    cube = crop2dfrom3d(cube, shape2, 2)

    # # store cube model to pt
    np.save(f"./shapes/3d/model2_{imgSize}.npy", cube)

    view1 = getView2dfrom3d(cube, 0, 0, 0)
    cv2.imwrite("view1.png", view1)
    view2 = getView2dfrom3d(cube, 90, 90, 90)
    cv2.imwrite("view2.png", view2)
    view3 = getView2dfrom3d(cube, 45, 45, 45)
    cv2.imwrite("view3.png", view3)

    setting_list = getSettingList2D()
    record2DVideo(cube, setting_list, f"./output/2d/2d_{imgSize}.gif")
