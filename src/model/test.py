# -*- coding: utf-8 -*-
import os
import sys
import cv2
import itertools
import skimage.color as color
import matplotlib.pyplot as plt
import scipy.ndimage.interpolation as sni
import argparse
import numpy as np
import skimage.io

sys.path.insert(0, '/media/deeplearning/Document_SSD/FrankLewis/DoctoralCodes/MultibandImageColorFusion/src/model')
import models


if __name__ == '__main__':

    # Set default params
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--mode', type=str, default='train', help="Choose train or eval")
    parser.add_argument('--data_file', type=str, default="../../data/processed/ColorFusion_128_data.h5",
                        help="Path to HDF5 containing the data")
    parser.add_argument('--training_mode', default="in_memory", type=str,
                        help=('Training mode. Choose in_memory to load all the data in memory and train.'
                              'Choose on_demand to load batches from disk at each step'))
    parser.add_argument('--batch_size', default=6, type=int, help='Batch size')
    parser.add_argument('--n_batch_per_epoch', default=50, type=int, help="Number of batches per epoch")
    parser.add_argument('--nb_epoch', default=51, type=int, help="Number of training epochs")
    parser.add_argument('--nb_resblocks', default=2, type=int, help="Number of residual blocks for simple model")
    parser.add_argument('--nb_neighbors', default=10, type=int, help="Number of nearest neighbors for soft encoding")
    parser.add_argument('--epoch', default=5, type=int, help="Epoch at which weights were saved for evaluation")
    parser.add_argument('--T', default=0.2, type=float,
                        help=("Temperature to change color balance in evaluation phase."
                              "If T = 1: desaturated. If T~0 vivid"))

    parser.add_argument('--img_dim', default=(3, 128, 128), type=tuple, help="set the training image size")
    parser.add_argument('--sub_name', default='', type=str, help="set the sub name of the model")
    parser.add_argument('--model_name', default='Colorization', type=str,
                        help="set the model name of the model, only use in test_class.py")
    args = parser.parse_args()

    d_params = {"data_file": args.data_file,
                 "batch_size": args.batch_size,
                 "n_batch_per_epoch": args.n_batch_per_epoch,
                 "nb_epoch": args.nb_epoch,
                 "nb_resblocks": args.nb_resblocks,
                 "training_mode": args.training_mode,
                 "nb_neighbors": args.nb_neighbors,
                 "epoch": args.epoch,
                 "T": 0.5,
                 "sub_name": "t0.5",
                 "img_dim": args.img_dim,
                 "model_name": 'Richard_Colorization'
                 }

    '''Load Model'''



    model_name = d_params['model_name']
    sub_name = d_params["sub_name"]

    # Load colorizer model
    if model_name == "Richard_Colorization":
        color_model = models.RichardImageColorizationModel().create_model(**d_params)
    elif model_name == "Richard_Colorization_V1":
        color_model = models.RichardImageColorizationModel_V1().create_model(**d_params)

    elif model_name == "Residual_Colorization":
        color_model = models.ResidualImageColorizationModel().create_model(**d_params)

    elif model_name == "Hypercolum_Colorization":
        color_model = models.HypercolumImageColorizationModel().create_model(**d_params)


    # Load weights
    weights_path = os.path.join('../../models/%s/%s/%s_weights_epoch50.h5' % (model_name, sub_name, model_name))
    color_model.load_weights(weights_path)

    directory = "../../data/raw/original"
    q_ab = np.load('../../data/processed/pts_in_hull.npy')  # load cluster centers
    nb_q = q_ab.shape[0]
    # put test images directory in list

    list_color = [directory + "/Color/" + file for file in sorted(os.listdir(directory+"/Color"))]
    list_II = [directory + "/II/" + file for file in sorted(os.listdir(directory + "/II"))]
    list_IR = [directory + "/IR/" + file for file in sorted(os.listdir(directory + "/IR"))]
    list_Vis = [directory + "/Vis/" + file for file in sorted(os.listdir(directory + "/Vis"))]

    name = 1
    for IIPath, IRPath, VisPath, ColorPath in itertools.izip(list_II, list_IR, list_Vis, list_color):

        img_II = cv2.imread(IIPath, 0) / 255.
        img_IR = cv2.imread(IRPath, 0) / 255.
        img_Vis = cv2.imread(VisPath, 0) / 255.

        img_color = cv2.imread(ColorPath) / 255.
        (H_orig, W_orig) = img_II.shape[:2]

        if False:
            (H_in, W_in) = (H_orig, W_orig)
        else:
            (H_in, W_in) = (480, 640)      # 输入到网络的尺寸必须是2的三次方倍,因为模型存在三次降采样
            img_II = cv2.resize(img_II, (W_in, H_in))     # cv2.resize 后的尺寸是先列后行,即（cols, rows）
            img_IR = cv2.resize(img_IR, (W_in, H_in))
            img_Vis = cv2.resize(img_Vis, (W_in, H_in))
            img_color = cv2.resize(img_color, (W_in, H_in))

        img_II = img_II.reshape((1, H_in, W_in, 1)).transpose(0, 3, 1, 2)
        img_IR = img_IR.reshape((1, H_in, W_in, 1)).transpose(0, 3, 1, 2)
        img_Vis = img_Vis.reshape((1, H_in, W_in, 1)).transpose(0, 3, 1, 2)

        img_merge = np.concatenate((img_II, img_IR, img_Vis), axis=1)

        '''
           进行NRL伪彩色融合并提取伪彩色融合图像亮度信息
        '''
        imgNRL = np.concatenate((img_IR, img_II, img_II), axis=1)   # 维度为(1, 3, H_in, W_in)
        imgNRL = imgNRL[0].transpose(1, 2, 0)
        imgNRL_lab = color.rgb2lab(imgNRL)
        imgNRL_l = imgNRL_lab[:, :, 0]
        # plt.figure(figsize=(20, 20))
        # plt.imshow(imgNRL)

        '''
           进行TNO伪彩色融合并提取伪彩色融合图像亮度信息
        '''




        # change the BGR to RGB mode
        img_color = img_color[:, :, ::-1]
        img_lab = color.rgb2lab(img_color)
        img_l = img_lab[:, :, 0]

        # img_l = np.expand_dims(img_l, axis=0)
        # img_l = np.expand_dims(img_l, axis=0)
        img_l = img_l.reshape((1, H_in, W_in, 1)).transpose(0, 3, 1, 2)   # 同上面两句实现同样的功能
        imgNRL_l = imgNRL_l.reshape((1, H_in, W_in, 1)).transpose(0, 3, 1, 2)

        # Predict
        # 同时预测亮度和色度信息
        # X_l = color_model.predict(img_merge)[1] * 100
        # X_colorized = color_model.predict(img_merge)[0]

        # 仅预测色度信息
        X_colorized = color_model.predict(img_merge)

        X_colorized = np.moveaxis(X_colorized, 1, 3)

        X_colorized = X_colorized.reshape((H_in*W_in, nb_q))

        X_colorized = q_ab[np.argmax(X_colorized, 1)]

        X_a = X_colorized[:, 0].reshape((1, 1, H_in, W_in))
        X_b = X_colorized[:, 1].reshape((1, 1, H_in, W_in))



        # # upsample to match size of original image L ,a , b
        # X_a = sni.zoom(X_a, (1, 1, 1. * H_orig / H_in, 1. * W_orig / W_in))
        # X_b = sni.zoom(X_b, (1, 1, 1. * H_orig / H_in, 1. * W_orig / W_in))
        # X_l = sni.zoom(X_l, (1, 1, 1. * H_orig / H_in, 1. * W_orig / W_in))
        #
        # img_l = sni.zoom(img_l, (1, 1, 1. * H_orig / H_in, 1. * W_orig / W_in))
        # img_II = sni.zoom(img_II, (1, 1, 1. * H_orig / H_in, 1. * W_orig / W_in))
        # img_IR = sni.zoom(img_IR, (1, 1, 1. * H_orig / H_in, 1. * W_orig / W_in))





        # Luminance information come from the false color image
        X_colorized_falsecolor = np.concatenate((imgNRL_l, X_a, X_b), axis=1).transpose(0, 2, 3, 1)


        # Luminance information come from the color image
        X_colorized_color = np.concatenate((img_l, X_a, X_b), axis=1).transpose(0, 2, 3, 1)

        # Luminance information come from the model
        # X_colorized = np.concatenate((X_l, X_a, X_b), axis=1).transpose(0, 2, 3, 1)

        # Luminance information come from the infrared image
        X_colorized_IR = np.concatenate((img_IR * 100, X_a, X_b), axis=1).transpose(0, 2, 3, 1)

        # Luminance information come from the LLL image
        X_colorized_II = np.concatenate((img_II * 100, X_a, X_b), axis=1).transpose(0, 2, 3, 1)

        # Luminance information come from the Vis image
        X_colorized_Vis = np.concatenate((img_Vis * 100, X_a, X_b), axis=1).transpose(0, 2, 3, 1)

        # Luminance information come from the average of IR and LLL image
        X_colorized_IIIR = np.concatenate(((img_II + img_IR) * 50, X_a, X_b), axis=1).transpose(0, 2, 3, 1)



        X_colorized = [np.expand_dims(color.lab2rgb(im), 0) for im in X_colorized_Vis]
        X_colorized = np.concatenate(X_colorized, 0).transpose(0, 3, 1, 2)
        arr = X_colorized[0]
        arr = arr.transpose(1, 2, 0)
        plt.figure(figsize=(20, 20))
        plt.imshow(arr)

        save_name = str(name)

        skimage.io.imsave('../../results/' + sub_name + '/' + save_name + '.jpg', arr)
        name += 1

        plt.show()
















