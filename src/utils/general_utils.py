# -*- coding: utf-8 -*-

import os
import numpy as np
from skimage import color
import matplotlib.pylab as plt


def remove_files(files):
    """
    Remove files from disk

    args: files (str or list) remove all files in 'files'
    """

    if isinstance(files, (list, tuple)):
        for f in files:
            if os.path.isfile(os.path.expanduser(f)):
                os.remove(f)
    elif isinstance(files, str):
        if os.path.isfile(os.path.expanduser(files)):
            os.remove(files)


def create_dir(dirs):
    """
    Create directory

    args: dirs (str or list) create all dirs in 'dirs'
    """

    if isinstance(dirs, (list, tuple)):
        for d in dirs:
            if not os.path.exists(os.path.expanduser(d)):
                os.makedirs(d)
    elif isinstance(dirs, str):
        if not os.path.exists(os.path.expanduser(dirs)):
            os.makedirs(dirs)


def setup_logging(model_name):

    model_dir = "../../models"
    # Output path where we store experiment log and weights
    model_dir = os.path.join(model_dir, model_name)

    fig_dir = "../../figures"

    # Create if it does not exist
    create_dir([model_dir, fig_dir])


def plot_batch_color(color_model, model_name, q_ab, X_batch_merge, X_batch_lab, batch_size, h, w, nb_q, epoch, sub_name):

    # Format X_colorized
    X_colorized = color_model.predict(X_batch_merge / 255.)
    X_colorized = np.moveaxis(X_colorized, 1, 3)


    X_colorized = X_colorized.reshape((batch_size * h * w, nb_q))
    X_colorized = q_ab[np.argmax(X_colorized, 1)]
    X_a = X_colorized[:, 0].reshape((batch_size, 1, h, w))
    X_b = X_colorized[:, 1].reshape((batch_size, 1, h, w))



    # 微光图像作为亮度信息
    X_batch_II = X_batch_merge[:, 0, :, :].reshape((batch_size, 1, h, w))

    X_colorized = np.concatenate((X_batch_II, X_a, X_b), axis=1).transpose(0, 2, 3, 1)

    X_colorized = [np.expand_dims(color.lab2rgb(im), 0) for im in X_colorized]
    X_colorized = np.concatenate(X_colorized, 0).transpose(0, 3, 1, 2)

    X_batch_color = [np.expand_dims(color.lab2rgb(im.transpose(1, 2, 0)), 0) for im in X_batch_lab]
    X_batch_color = np.concatenate(X_batch_color, 0).transpose(0, 3, 1, 2)

    list_img = []
    for i, img in enumerate(X_colorized[:min(32, batch_size)]):
        X_LLL = np.reshape(X_batch_merge[i][0], (1, 128, 128))
        X_IR = np.reshape(X_batch_merge[i][1], (1, 128, 128))

        # 同时预测亮度和色度信息
        # arr = np.concatenate([X_batch_color[i], np.repeat(X_LLL / 255., 3, axis=0), np.repeat(X_IR / 255., 3, axis=0),
        #                       np.repeat(X_batch_l[i] / 100., 3, axis=0), img], axis=2)

        # 仅预测色度信息
        arr = np.concatenate([X_batch_color[i], np.repeat(X_LLL / 255., 3, axis=0), np.repeat(X_IR / 255., 3, axis=0), img], axis=2)
        list_img.append(arr)

    plt.figure(figsize=(20,20))
    list_img = [np.concatenate(list_img[4 * i: 4 * (i + 1)], axis=2) for i in range(len(list_img) / 4)]
    arr = np.concatenate(list_img, axis=1)
    plt.imshow(arr.transpose(1,2,0))
    ax = plt.gca()
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    plt.tight_layout()

    weights_path = os.path.join("../../figures/%s/%s/" % (model_name, sub_name))
    if not os.path.exists(weights_path):
        os.makedirs(weights_path)

    plt.savefig(weights_path + "fig_epoch%s.png" % epoch)
    plt.clf()
    plt.close()


def plot_batch(color_model, model_name, q_ab, X_batch_merge, X_batch_lab, batch_size, h, w, nb_q, epoch, sub_name):

    # Format X_colorized
    X_colorized = color_model.predict(X_batch_merge / 255.)[0]
    X_batch_l = color_model.predict(X_batch_merge / 255.)[1] * 100

    X_colorized = np.moveaxis(X_colorized, 1, 3)


    X_colorized = X_colorized.reshape((batch_size * h * w, nb_q))
    X_colorized = q_ab[np.argmax(X_colorized, 1)]
    X_a = X_colorized[:, 0].reshape((batch_size, 1, h, w))
    X_b = X_colorized[:, 1].reshape((batch_size, 1, h, w))
    X_colorized = np.concatenate((X_batch_l, X_a, X_b), axis=1).transpose(0, 2, 3, 1)
    X_colorized = [np.expand_dims(color.lab2rgb(im), 0) for im in X_colorized]
    X_colorized = np.concatenate(X_colorized, 0).transpose(0, 3, 1, 2)

    X_batch_color = [np.expand_dims(color.lab2rgb(im.transpose(1, 2, 0)), 0) for im in X_batch_lab]
    X_batch_color = np.concatenate(X_batch_color, 0).transpose(0, 3, 1, 2)

    list_img = []
    for i, img in enumerate(X_colorized[:min(32, batch_size)]):
        X_LLL = np.reshape(X_batch_merge[i][0], (1, 128, 128))
        X_IR = np.reshape(X_batch_merge[i][1], (1, 128, 128))
        arr = np.concatenate([X_batch_color[i], np.repeat(X_LLL / 255., 3, axis=0), np.repeat(X_IR / 255., 3, axis=0),
                              np.repeat(X_batch_l[i] / 100., 3, axis=0), img], axis=2)
        list_img.append(arr)

    plt.figure(figsize=(20,20))
    list_img = [np.concatenate(list_img[4 * i: 4 * (i + 1)], axis=2) for i in range(len(list_img) / 4)]
    arr = np.concatenate(list_img, axis=1)
    plt.imshow(arr.transpose(1,2,0))
    ax = plt.gca()
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    plt.tight_layout()

    weights_path = os.path.join("../../figures/%s/%s/" % (model_name, sub_name))
    if not os.path.exists(weights_path):
        os.makedirs(weights_path)

    plt.savefig(weights_path + "fig_epoch%s.png" % epoch)
    plt.clf()
    plt.close()