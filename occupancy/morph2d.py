import numpy as np
import cv2
import scipy.ndimage
from PIL import Image
import tqdm

DEBUG = True


def getShape2d(type, size=100, imgSize=200, skeleton=False):
    if type == 0:
        # rectangle
        shape = np.zeros((size, size))
        border = int(size * 0.2)
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
        radius = size * 0.3
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
    # rotate cube
    cube = scipy.ndimage.rotate(cube, yaw, axes=(1, 2), reshape=False)
    cube = scipy.ndimage.rotate(cube, pitch, axes=(0, 2), reshape=False)
    cube = scipy.ndimage.rotate(cube, roll, axes=(0, 1), reshape=False)
    # get view
    view = np.max(cube, axis=0)
    return view


def getSettingList2D():
    setting_list = []
    for angle in range(0, 360, 10):
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


if __name__ == "__main__":
    size = 80
    imgSize = 100
    b_r = int((imgSize - size) / 2)
    cube = np.zeros((imgSize, imgSize, imgSize))
    # cube = np.ones((size, size, size)) * 255
    cube[b_r : b_r + size, b_r : b_r + size, b_r : b_r + size] = 255
    shape1 = getShape2d(0, size, imgSize, True)
    shape2 = getShape2d(1, size, imgSize, True)
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
    record2DVideo(cube, setting_list, "./output/2d/2d_test.gif")
