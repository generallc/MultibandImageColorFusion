# -*- coding: utf-8 -*-
import os
import sys
import h5py
import time
import glob
import numpy as np
import sklearn.neighbors as nn
import keras.backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Activation, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.merge import Average
from keras.utils import generic_utils
from keras.optimizers import SGD, Adam
from keras.utils.visualize_util import plot
# Utils
sys.path.insert(0, '/media/generallc/DoctoralResearch/DoctoralCodes/LLLInfraredColorful_threeband/src/utils')
import batch_utils
import general_utils



def categorical_crossentropy_color(prior_factor):
    prior_factor = K.variable(prior_factor)
    def loss(y_true, y_pred):
        # Flatten
        y_true = K.reshape(y_true, (-1, 313))

        # the prediction is "th" mode So transpose dimensions, put the filter number in the last dimension
        y_pred = K.permute_dimensions(y_pred, [0, 2, 3, 1])
        y_pred = K.reshape(y_pred, (-1, 313))

        # normalized exponential function, calculate the probability distribution of Chroma information
        y_pred = K.softmax(y_pred)

        # Returns the index of the maximum value along an axis
        idx_max = K.argmax(y_true, axis=1)

        # Retrieves the elements of indices indices in the tensor reference
        weights = K.gather(prior_factor, idx_max)

        weights = K.reshape(weights, (y_true.shape[0], 1))
        weights = K.concatenate([weights] * 313, axis=1)

        # multiply y_true by weights
        y_true = y_true * weights

        cross_ent = K.categorical_crossentropy(y_pred, y_true)
        cross_ent = K.mean(cross_ent, axis=-1)
        return cross_ent
    return loss


def convolutional_block(x, block_idx, nb_filter, nb_conv, stride):

    # 1st conv
    for i in range(nb_conv):
        name = "block%s_conv2D_%s" % (block_idx, i)
        if i < nb_conv - 1:
            x = Conv2D(nb_filter, (3, 3), name=name, padding="same")(x)
            x = BatchNormalization(axis=1)(x)
            x = Activation("relu")(x)
        else:
            x = Conv2D(nb_filter, (3, 3), name=name, strides=stride, padding="same")(x)
            x = BatchNormalization(axis=1)(x)
            x = Activation("relu")(x)

    return x


def atrous_block(x, block_idx, nb_filter, nb_conv):

    # 1st conv
    for i in range(nb_conv):
        name = "block%s_conv2D_%s" % (block_idx, i)
        x = Conv2D(nb_filter, (3, 3), name=name, padding="same", dilation_rate=(2, 2))(x)
        x = BatchNormalization(axis=1)(x)

        x = Activation("relu")(x)

    return x


class BaseImageColorizationModel(object):

    def __init__(self, model_name):
        """
        Base model to provide a standard interface of adding image Colorization models
        """
        self.model = None       # type: Model
        self.model_name = model_name
        self.weight_path = None

    def create_model(self, **kwargs):
        """
        Subclass dependent implementation.
        """
        img_dim = kwargs["img_dim"]
        channels, width, height = img_dim
        if K.image_dim_ordering() == "th":
            shape = (channels, None, None)
        else:
            shape = (None, None, channels)

        init = Input(shape=shape, name='input')
        return init

    def fit(self, **kwargs):
        """
        Standard method to train any of the models.
        """
        # Roll out the parameters
        batch_size = kwargs["batch_size"]
        n_batch_per_epoch = kwargs["n_batch_per_epoch"]
        nb_epochs = kwargs["nb_epoch"]
        data_file = kwargs["data_file"]
        nb_neighbors = kwargs["nb_neighbors"]
        training_mode = kwargs["training_mode"]
        epoch_size = n_batch_per_epoch * batch_size
        img_dim = kwargs["img_dim"]
        sub_name = kwargs["sub_name"]
        channels, width, height = img_dim

        # extract the patches size from the file name
        img_size = int(os.path.basename(data_file).split("_")[1])

        # Setup directories to save model, architecture etc
        general_utils.setup_logging(self.model_name)

        # Remove possible previous figures to avoid confusion
        for f in glob.glob("../../figures/*.png"):
            os.remove(f)

        # Load and rescale data
        if training_mode == "in_memory":
            with h5py.File(data_file, "r") as hf:
                color_lab = hf["training_lab_data"][:]
                LLLIR_merge = hf["training_merge_data"][:]

        # Load the array of quantized ab value
        q_ab = np.load("../../data/processed/pts_in_hull.npy")
        nb_q = q_ab.shape[0]

        # Fit a NN to q_ab
        nn_finder = nn.NearestNeighbors(n_neighbors=nb_neighbors, algorithm='ball_tree').fit(q_ab)

        # Load the color prior factor that encourages rare colors
        prior_factor = np.load("../../data/processed/ColorFusion_%s_prior_factor.npy" % img_size)

        # Create a batch generator for the color data
        DataGen = batch_utils.DataGenerator(data_file, batch_size=batch_size, dset="training")

        if self.model:
            self.create_model(**kwargs)

        print("Training model : %s" % (self.__class__.__name__))

        for epoch in range(nb_epochs):

            # Initialize progbar and batch counter
            progbar = generic_utils.Progbar(epoch_size)
            batch_counter = 1
            start = time.time()

            # Choose Batch Generation mode
            if training_mode == "in_memory":
                BatchGen = DataGen.gen_batch_in_memory(color_lab, LLLIR_merge, nn_finder, nb_q)
            else:
                BatchGen = DataGen.gen_batch(nn_finder, nb_q, prior_factor)

            for batch in BatchGen:

                X_batch_merge, X_batch_lab, Y_batch, X_batch_l, X_batch_ab= batch

                X_batch_merge = X_batch_merge.astype('float64')

                # 同时预测亮度信息和色度信息
                # train_loss = self.model.train_on_batch(X_batch_merge / 255., {'color_output': Y_batch, 'gray_output': X_batch_l / 100.})

                #仅预测色度信息
                train_loss = self.model.train_on_batch(X_batch_merge / 255., Y_batch)

                batch_counter += 1

                # progbar.add(batch_size, values=[("color_loss", train_loss[1]), ("gray_loss", train_loss[2])])

                # 仅预测色度信息
                progbar.add(batch_size, values=[("color_loss", train_loss)])

                if batch_counter >= n_batch_per_epoch:
                    break

            print("")
            print('Epoch %s/%s, Time: %s' % (epoch + 1, nb_epochs, time.time() - start))

            # 同时预测亮度信息和色度信息
            # general_utils.plot_batch(self.model, self.model_name, q_ab, X_batch_merge, X_batch_lab,
            #                          batch_size, height, width, nb_q, epoch, sub_name)
            # 仅预测色度信息
            general_utils.plot_batch_color(self.model, self.model_name, q_ab, X_batch_merge, X_batch_lab,
                                           batch_size, height, width, nb_q, epoch, sub_name)

            # Save weights every 5 epoch
            if epoch % 5 == 0:
                weights_path = os.path.join('../../models/%s/%s/' % (self.model_name, sub_name))
                if not os.path.exists(weights_path):
                    os.makedirs(weights_path)
                self.model.save_weights(weights_path + "%s_weights_epoch%s.h5" % (self.model_name, epoch), overwrite=True)
        return self.model


'''同colorful image colorization 完全一样的结构'''


class RichardImageColorizationModel(BaseImageColorizationModel):

    def __init__(self):
        super(RichardImageColorizationModel, self).__init__("Richard_Colorization")
        self.nb_resblocks = 2

    def create_model(self, **kwargs):
        """
            Creates a model to be used to scale images of specific height and width.
        """
        sub_name = kwargs["sub_name"]

        # Load the array of quantized ab value
        q_ab = np.load("../../data/processed/pts_in_hull.npy")
        nb_q = q_ab.shape[0]

        init = super(RichardImageColorizationModel, self).create_model(**kwargs)

        # Convolutional blocks parameters  The 1,2,3,4,7 block
        list_filter_size = [64, 128, 256, 512, 256]
        list_block_size = [2, 2, 3, 3, 3]
        stride = [(2, 2), (2, 2), (2, 2), (1, 1), (1, 1)]   # decrease the spatial resolution through the (2,2) stride

        # A trous blocks parameters
        list_filter_size_atrous = [512, 512]
        list_block_size_atrous = [3, 3]

        block_idx = 0

        # First block
        f, b, s = list_filter_size[0], list_block_size[0], stride[0]
        x = convolutional_block(init, block_idx, f, b, s)
        block_idx += 1


        # Next blocks(the 2,3,4 blocks)
        for f, b, s in zip(list_filter_size[1:-1], list_block_size[1:-1], stride[1:-1]):
            x = convolutional_block(x, block_idx, f, b, s)
            block_idx += 1


        # Atrous blocks(the 5,6 blocks)
        for idx, (f, b) in enumerate(zip(list_filter_size_atrous, list_block_size_atrous)):
            x = atrous_block(x, block_idx, f, b)
            block_idx += 1

        # Block 7
        f, b, s = list_filter_size[-1], list_block_size[-1], stride[-1]
        x = convolutional_block(x, block_idx, f, b, s)
        block_idx += 1


        # Block 8
        # Not using Deconvolution at the moment
        # x = Deconvolution2D(256, 2, 2,
        #                     output_shape=(None, 256, current_h * 2, current_w * 2),
        #                     subsample=(2, 2),
        #                     border_mode="valid")(x)
        x = UpSampling2D(size=(2, 2), name="upsampling2d")(x)

        x = convolutional_block(x, block_idx, 128, 2, (1, 1))
        block_idx += 1


        # gray upsample

        y = UpSampling2D(size=(2, 2), name="upsampling2d_1")(x)

        y = convolutional_block(y, block_idx, 128, 1, (1, 1))
        block_idx += 1

        y = UpSampling2D(size=(2, 2), name="upsampling2d_2")(y)

        y = convolutional_block(y, block_idx, 64, 1, (1, 1))
        block_idx += 1

        # Final output

        gray_output = Conv2D(1, (1, 1), name="gray_output", padding="same")(y)

        x = Conv2D(nb_q, (1, 1), name="conv2d_final", padding="same")(x)
        color_output = UpSampling2D(size=(4, 4), dim_ordering="th", name='color_output')(x)

        # Build model
        prior_factor = np.load("../../data/processed/ColorFusion_%s_prior_factor.npy" % 128)

        # 同时预测亮度信息和色度信息,即多任务输出
        # model = Model(input=[init], output=[color_output, gray_output], name=self.model_name)

        # 仅预测色度信息
        model = Model(input=[init], output=[color_output], name=self.model_name)

        sgd = SGD(lr=0.0005, decay=0.5e-3, momentum=0.9, nesterov=True)

        # 同时预测亮度信息和色度信息,即多任务输出
        # model.compile(loss={'color_output': categorical_crossentropy_color(prior_factor), 'gray_output': 'mae'},
        #               loss_weights=[1., 1.], optimizer=sgd)

        # 仅预测色度信息
        model.compile(loss=categorical_crossentropy_color(prior_factor), optimizer=sgd)

        self.model = model
        model.summary()

        plot(model, to_file='../../figures/%s_%s.png' % (self.model_name, sub_name), show_shapes=True, show_layer_names=True)
        return model

    def fit(self, **kwargs):
        return super(RichardImageColorizationModel, self).fit(**kwargs)


'''自己设计的结构,对于亮度图像预测网络包含跳跃结构'''


class RichardImageColorizationModel_V1(BaseImageColorizationModel):

    def __init__(self):
        super(RichardImageColorizationModel_V1, self).__init__("Richard_Colorization_V1")
        self.nb_resblocks = 2

    def create_model(self, **kwargs):
        """
            Creates a model to be used to scale images of specific height and width.
        """
        sub_name = kwargs["sub_name"]


        # Load the array of quantized ab value
        q_ab = np.load("../../data/processed/pts_in_hull.npy")
        nb_q = q_ab.shape[0]

        init = super(RichardImageColorizationModel_V1, self).create_model(**kwargs)

        level1_1 = Conv2D(64, (3, 3), activation='relu', padding='same', strides=(2, 2), kernel_initializer='he_normal',
                          name='level1_1')(init)
        level1_1 = BatchNormalization(axis=1)(level1_1)

        level2_1 = Conv2D(128, (3, 3), activation='relu', padding='same', strides=(2, 2), kernel_initializer='he_normal',
                          name='level2_1')(level1_1)
        level2_1 = BatchNormalization(axis=1)(level2_1)

        level3_1 = Conv2D(256, (3, 3), activation='relu', padding='same', strides=(2, 2), kernel_initializer='he_normal',
                          name='level3_1')(level2_1)
        level3_1 = BatchNormalization(axis=1)(level3_1)

        level5_1 = Conv2D(256, (3, 3), activation='relu', padding='same', dilation_rate=(2, 2), kernel_initializer='he_normal')(level3_1)
        level5_1 = BatchNormalization(axis=1)(level5_1)

        level5_1 = Conv2D(256, (3, 3), activation='relu', padding='same', dilation_rate=(2, 2), kernel_initializer='he_normal')(level5_1)
        level5_1 = BatchNormalization(axis=1)(level5_1)

        level5_1 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(level5_1)
        level5_1 = BatchNormalization(axis=1)(level5_1)



        # gray upsample

        level3_2 = UpSampling2D(size=(2, 2), name="upsampling2d_2")(level5_1)
        level3_2 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(level3_2)

        level3 = Average()([level2_1, level3_2])



        level2_2 = UpSampling2D(size=(2, 2), name="upsampling2d_3")(level3)
        level2_2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='level2_2')(level2_2)

        level2 = Average()([level1_1, level2_2])

        level1_2 = UpSampling2D(size=(2, 2), name="upsampling2d_4")(level2)
        level1_2 = Conv2D(3, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='level1_2')(level1_2)

        level1 = Average()([init, level1_2])

        gray_output = Conv2D(1, (3, 3), name="gray_output", padding="same")(level1)


        # Final conv
        Final_output = Conv2D(nb_q, (3, 3), name="final_conv2D", padding="same")(level3_2)

        color_output = UpSampling2D(size=(4, 4), dim_ordering="th", name='color_output')(Final_output)


        prior_factor = np.load("../../data/processed/ColorFusion_%s_prior_factor.npy" % 128)



        # Build model

        # 同时预测亮度信息和色度信息,即多任务输出
        # model = Model(input=[init], output=[color_output, gray_output], name=self.model_name)

        # 仅预测色度信息
        model = Model(input=[init], output=[color_output], name=self.model_name)

        sgd = SGD(lr=0.0005, decay=0.5e-3, momentum=0.9, nesterov=True)

        # 同时预测亮度信息和色度信息,即多任务输出
        # model.compile(loss={'color_output': categorical_crossentropy_color(prior_factor), 'gray_output': 'mae'},
        #               loss_weights=[1., 1.], optimizer=sgd)

        # 仅预测色度信息
        model.compile(loss=categorical_crossentropy_color(prior_factor), optimizer=sgd)

        self.model = model
        model.summary()

        plot(model, to_file='../../figures/%s_%s.png' % (self.model_name, sub_name), show_shapes=True, show_layer_names=True)
        return model

    def fit(self, **kwargs):
        return super(RichardImageColorizationModel_V1, self).fit(**kwargs)






