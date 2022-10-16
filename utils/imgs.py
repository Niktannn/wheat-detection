import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def img_reshape(img, d):
    img = Image.open(img).convert('RGB')
    img = img.resize((d, d))
    img = np.asarray(img)
    return img


def image_grid(array, ncols=2):
    index, height, width, channels = array.shape
    nrows = index // ncols
    img_grid = (array.reshape(nrows, ncols, height, width, channels)
                .swapaxes(1, 2)
                .reshape(height * nrows, width * ncols, -1))

    return img_grid


def show_imgs(show_dir, show_n=10, n_col=2):
    images = [os.path.join(show_dir, image) for image in os.listdir(show_dir)]
    show_n = min(show_n, len(images))
    images = images[:show_n]
    img_arr = []

    for image in images:
        img_arr.append(img_reshape(image, 800))

    result = image_grid(np.array(img_arr), ncols=n_col)
    fig = plt.figure(figsize=(40, 40))
    plt.imshow(result)