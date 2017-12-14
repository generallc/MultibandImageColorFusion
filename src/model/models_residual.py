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
from keras.layers.convolutional import Conv2D, UpSampling2D,Conv2DTranspose
from keras.layers.merge import Average, Concatenate
from keras.utils import generic_utils
from keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta, Nadam
from keras.utils.vis_utils import plot_model
from keras_contrib.layers import SubPixelUpscaling
from keras.utils.generic_utils import get_custom_objects

# Utils
sys.path.insert(0, '/media/deeplearning/Document_SSD/FrankLewis/DoctoralCodes/MultibandImageColorFusion/src/utils')
import batch_utils
import general_utils

# 主要是论文使用的Residual网络结构

class SubPixel(SubPixelUpscaling):
    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            b, k, r, c = input_shape
            if r is None:
                return b, k // (self.scale_factor ** 2), None, None
            else:
                return b, k // (self.scale_factor ** 2), r * self.scale_factor, c * self.scale_factor
        else:
            b, r, c, k = input_shape
            return b, r * self.scale_factor, c * self.scale_factor, k // (self.scale_factor ** 2)


get_custom_objects().update({'SubPixel': SubPixel})



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
        epoch_size = int(n_batch_per_epoch * batch_size * 1.1)
        img_dim = kwargs["img_dim"]
        sub_name = kwargs["sub_name"]
        history = kwargs["history"]
        channels, width, height = img_dim

        # extract the patches size from the file name
        img_size = int(os.path.basename(data_file).split("_")[1])

        # Setup directories to save model, architecture etc
        general_utils.setup_logging(self.model_name)

        # Remove possible previous figures to avoid confusion
        for f in glob.glob("../../figures/*.png"):
            os.remove(f)

        # Load and rescale training data
        if training_mode == "in_memory":
            with h5py.File(data_file, "r") as hf:
                color_lab = hf["training_lab_data"][:]
                LLLIR_merge = hf["training_merge_data"][:]

        # Load and rescale validation data
        if training_mode == "in_memory":
            with h5py.File(data_file, "r") as hf:
                val_color_lab = hf["validation_lab_data"][:]
                val_LLLIR_merge = hf["validation_merge_data"][:]



        # Load the array of quantized ab value
        q_ab = np.load("../../data/processed/pts_in_hull.npy")
        nb_q = q_ab.shape[0]

        # Fit a NN to q_ab
        nn_finder = nn.NearestNeighbors(n_neighbors=nb_neighbors, algorithm='ball_tree').fit(q_ab)

        # Load the color prior factor that encourages rare colors
        prior_factor = np.load("../../data/processed/ColorFusion_%s_prior_factor.npy" % img_size)

        # Create a batch generator for the color data
        DataGen = batch_utils.DataGenerator(data_file, batch_size=batch_size, dset="training")

        # Create a batch generator for the validation color data

        ValDataGen = batch_utils.DataGenerator(data_file, batch_size=batch_size, dset="validation")



        if self.model:
            self.create_model(**kwargs)

        print("Training model : %s" % (self.__class__.__name__))

        loss_data = {"color_loss": [], "val_color_loss": [], "batch_color_loss": [], "gray_loss": [], "val_gray_loss": [], "batch_gray_loss": []}



        for epoch in range(nb_epochs):

            # Initialize progbar and batch counter
            progbar = generic_utils.Progbar(epoch_size)
            batch_counter = 1
            val_batch_counter = 1
            start = time.time()

            # Choose Batch Generation mode
            if training_mode == "in_memory":
                BatchGen = DataGen.gen_batch_in_memory(color_lab, LLLIR_merge, nn_finder, nb_q)
            else:
                BatchGen = DataGen.gen_batch(nn_finder, nb_q, prior_factor)

            # Choose Validation Batch Generation mode
            if training_mode == "in_memory":
                ValBatchGen = ValDataGen.gen_batch_in_memory(val_color_lab, val_LLLIR_merge, nn_finder, nb_q)
            else:
                ValBatchGen = ValDataGen.gen_batch(nn_finder, nb_q, prior_factor)

            color_loss_temp = np.array([])
            val_color_loss_temp = np.array([])

            gray_loss_temp = np.array([])
            val_gray_loss_temp = np.array([])


            for batch in BatchGen:

                X_batch_merge, X_batch_lab, Y_batch, X_batch_l, X_batch_ab= batch

                X_batch_merge = X_batch_merge.astype('float64')

                # 同时预测亮度信息和色度信息
                # train_loss = self.model.train_on_batch(X_batch_merge / 255., {'color_output': Y_batch, 'gray_output': X_batch_l / 100.})
                # batch_counter += 1
                # progbar.add(batch_size, values=[("color_loss", train_loss[1]), ("gray_loss", train_loss[2])])



                #仅预测色度信息
                train_loss = self.model.train_on_batch(X_batch_merge / 255., Y_batch)
                batch_counter += 1
                progbar.add(batch_size, values=[("color_loss", train_loss)])


                loss_data["batch_color_loss"].append(float(train_loss))
                # loss_data["batch_gray_loss"].append(float(train_loss[2]))
                color_loss_temp = np.append(color_loss_temp, train_loss)
                # gray_loss_temp = np.append(gray_loss_temp, train_loss[2])

                if batch_counter >= n_batch_per_epoch:
                    loss_data["color_loss"].append(np.average(color_loss_temp))
                    # loss_data["gray_loss"].append(np.average(gray_loss_temp))
                    break

            for valbatch in ValBatchGen:

                X_batch_merge, X_batch_lab, Y_batch, X_batch_l, X_batch_ab= valbatch

                X_batch_merge = X_batch_merge.astype('float64')

                # 同时预测亮度信息和色度信息
                # test_loss = self.model.test_on_batch(X_batch_merge / 255., {'color_output': Y_batch, 'gray_output': X_batch_l / 100.})
                # val_batch_counter += 1
                # progbar.add(batch_size, values=[("val_color_loss", test_loss[1]), ("val_gray_loss", test_loss[2])])

                #仅预测色度信息
                test_loss = self.model.test_on_batch(X_batch_merge / 255., Y_batch)
                val_batch_counter += 1
                progbar.add(batch_size, values=[("val_color_loss", test_loss)])


                val_color_loss_temp = np.append(val_color_loss_temp, test_loss)
                # val_gray_loss_temp = np.append(val_gray_loss_temp, test_loss[2])
                if val_batch_counter >= n_batch_per_epoch/10:
                    loss_data["val_color_loss"].append(np.average(val_color_loss_temp))
                    # loss_data["val_gray_loss"].append(np.average(val_gray_loss_temp))
                    break

            with open(history, "w") as f:
                f.write(str(loss_data))
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


'''Automatic Colorization设计的结构,对于亮度图像预测网络包含跳跃结构'''


class ResidualImageColorizationModel(BaseImageColorizationModel):

    def __init__(self):
        super(ResidualImageColorizationModel, self).__init__("Residual_Colorization")
        self.nb_resblocks = 2

    def create_model(self, **kwargs):
        """Residual_Colorization
            Creates a model to be used to scale images of specific height and width.
        """
        sub_name = kwargs["sub_name"]
        lr = kwargs["lr"]

        # Load the array of quantized ab value
        q_ab = np.load("../../data/processed/pts_in_hull.npy")
        nb_q = q_ab.shape[0]

        init = super(ResidualImageColorizationModel, self).create_model(**kwargs)

        level1_1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(init)

        level2_1 = Conv2D(64, (3, 3), activation='relu', padding='same', strides=(2, 2), kernel_initializer='he_normal')(level1_1)
        level2_1 = BatchNormalization(axis=1)(level2_1)

        level3_1 = Conv2D(128, (3, 3), activation='relu', padding='same', strides=(2, 2), kernel_initializer='he_normal')(level2_1)
        level3_1 = BatchNormalization(axis=1)(level3_1)

        level4_1 = Conv2D(256, (3, 3), activation='relu', padding='same', strides=(2, 2), kernel_initializer='he_normal')(level3_1)
        level4_1 = BatchNormalization(axis=1)(level4_1)

        level4_2 = Conv2D(256, (3, 3), activation='relu', padding='same', dilation_rate=(2, 2), kernel_initializer='he_normal')(level4_1)
        level4_2 = BatchNormalization(axis=1)(level4_2)

        level4_2 = Conv2D(256, (3, 3), activation='relu', padding='same', dilation_rate=(2, 2),
                          kernel_initializer='he_normal')(level4_2)
        level4_2 = BatchNormalization(axis=1)(level4_2)

        level4_2 = Conv2D(256, (3, 3), activation='relu', padding='same', dilation_rate=(2, 2),
                          kernel_initializer='he_normal')(level4_2)
        level4_2 = BatchNormalization(axis=1)(level4_2)

        # upsample

        level3_2 = UpSampling2D(size=(2, 2), name="upsampling2d_1")(level4_2)
        level3_2 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(level3_2)
        # level3_2 = BatchNormalization(axis=1)(level3_2)

        level3 = Average()([level3_1, level3_2])

        level2_2 = UpSampling2D(size=(2, 2), name="upsampling2d_2")(level3)
        level2_2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='level2_2')(level2_2)
        # level2_2 = BatchNormalization(axis=1)(level2_2)

        level2 = Average()([level2_1, level2_2])


        # Final conv
        Final_output = Conv2D(nb_q, (3, 3), name="final_conv2D", padding="same")(level2)

        color_output = UpSampling2D(size=(2, 2), dim_ordering="th", name='color_output')(Final_output)


        prior_factor = np.load("../../data/processed/ColorFusion_%s_prior_factor.npy" % 128)

        # sgd = SGD(lr=0.0005, decay=0.4e-5, momentum=0.9, nesterov=True)
        adam = Adam(lr=lr)  # 默认为0.001


        # Build model

        # 同时预测亮度信息和色度信息,即多任务输出
        # model = Model(input=[init], output=[color_output, gray_output], name=self.model_name)
        # model.compile(loss={'color_output': categorical_crossentropy_color(prior_factor), 'gray_output': 'mae'},
        #               loss_weights=[1., 1.], optimizer=adam)

        # 仅预测色度信息
        model = Model(input=[init], output=[color_output], name=self.model_name)
        model.compile(loss=categorical_crossentropy_color(prior_factor), optimizer=adam)

        self.model = model
        model.summary()

        plot_model(model, to_file='../../figures/%s_%s.png' % (self.model_name, sub_name), show_shapes=True, show_layer_names=True)
        return model

    def fit(self, **kwargs):
        return super(ResidualImageColorizationModel, self).fit(**kwargs)





class ResidualImageColorizationModel_subpixel(BaseImageColorizationModel):

    def __init__(self):
        super(ResidualImageColorizationModel_subpixel, self).__init__("Residual_Colorization_subpixel")
        self.nb_resblocks = 2

    def create_model(self, **kwargs):
        """Residual_Colorization
            Creates a model to be used to scale images of specific height and width.
        """
        sub_name = kwargs["sub_name"]
        lr = kwargs["lr"]

        # Load the array of quantized ab value
        q_ab = np.load("../../data/processed/pts_in_hull.npy")
        nb_q = q_ab.shape[0]

        init = super(ResidualImageColorizationModel_subpixel, self).create_model(**kwargs)

        level1_1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(init)

        level2_1 = Conv2D(64, (3, 3), activation='relu', padding='same', strides=(2, 2), kernel_initializer='he_normal')(level1_1)
        level2_1 = BatchNormalization(axis=1)(level2_1)

        level3_1 = Conv2D(128, (3, 3), activation='relu', padding='same', strides=(2, 2), kernel_initializer='he_normal')(level2_1)
        level3_1 = BatchNormalization(axis=1)(level3_1)

        level4_1 = Conv2D(256, (3, 3), activation='relu', padding='same', strides=(2, 2), kernel_initializer='he_normal')(level3_1)
        level4_1 = BatchNormalization(axis=1)(level4_1)

        level4_2 = Conv2D(256, (3, 3), activation='relu', padding='same', dilation_rate=(2, 2), kernel_initializer='he_normal')(level4_1)
        level4_2 = BatchNormalization(axis=1)(level4_2)

        level4_2 = Conv2D(256, (3, 3), activation='relu', padding='same', dilation_rate=(2, 2),
                          kernel_initializer='he_normal')(level4_2)
        level4_2 = BatchNormalization(axis=1)(level4_2)

        level4_2 = Conv2D(256, (3, 3), activation='relu', padding='same', dilation_rate=(2, 2),
                          kernel_initializer='he_normal')(level4_2)
        level4_2 = BatchNormalization(axis=1)(level4_2)

        # upsample

        level3_2 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(level4_2)
        level3_2 = SubPixel(scale_factor=2)(level3_2)

        # level3_2 = BatchNormalization(axis=1)(level3_2)

        level3 = Average()([level3_1, level3_2])

        level2_2 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(level3)
        level2_2 = SubPixel(scale_factor=2)(level2_2)

        # level2_2 = BatchNormalization(axis=1)(level2_2)

        level2 = Average()([level2_1, level2_2])


        # Final conv
        Final_output = Conv2D(nb_q, (3, 3), name="final_conv2D", padding="same")(level2)

        color_output = UpSampling2D(size=(2, 2), dim_ordering="th", name='color_output')(Final_output)


        prior_factor = np.load("../../data/processed/ColorFusion_%s_prior_factor.npy" % 128)

        # sgd = SGD(lr=0.0005, decay=0.4e-5, momentum=0.9, nesterov=True)
        adam = Adam(lr=lr)  # 默认为0.001


        # Build model

        # 同时预测亮度信息和色度信息,即多任务输出
        # model = Model(input=[init], output=[color_output, gray_output], name=self.model_name)
        # model.compile(loss={'color_output': categorical_crossentropy_color(prior_factor), 'gray_output': 'mae'},
        #               loss_weights=[1., 1.], optimizer=adam)

        # 仅预测色度信息
        model = Model(input=[init], output=[color_output], name=self.model_name)
        model.compile(loss=categorical_crossentropy_color(prior_factor), optimizer=adam)

        self.model = model
        model.summary()

        plot_model(model, to_file='../../figures/%s_%s.png' % (self.model_name, sub_name), show_shapes=True, show_layer_names=True)
        return model

    def fit(self, **kwargs):
        return super(ResidualImageColorizationModel_subpixel, self).fit(**kwargs)


class ResidualImageColorizationModel_trans(BaseImageColorizationModel):

    def __init__(self):
        super(ResidualImageColorizationModel_trans, self).__init__("Residual_Colorization_trans")
        self.nb_resblocks = 2

    def create_model(self, **kwargs):
        """Residual_Colorization
            Creates a model to be used to scale images of specific height and width.
        """
        sub_name = kwargs["sub_name"]
        lr = kwargs["lr"]

        # Load the array of quantized ab value
        q_ab = np.load("../../data/processed/pts_in_hull.npy")
        nb_q = q_ab.shape[0]

        init = super(ResidualImageColorizationModel_trans, self).create_model(**kwargs)

        level1_1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(init)

        level2_1 = Conv2D(64, (3, 3), activation='relu', padding='same', strides=(2, 2), kernel_initializer='he_normal')(level1_1)
        level2_1 = BatchNormalization(axis=1)(level2_1)

        level3_1 = Conv2D(128, (3, 3), activation='relu', padding='same', strides=(2, 2), kernel_initializer='he_normal')(level2_1)
        level3_1 = BatchNormalization(axis=1)(level3_1)

        level4_1 = Conv2D(256, (3, 3), activation='relu', padding='same', strides=(2, 2), kernel_initializer='he_normal')(level3_1)
        level4_1 = BatchNormalization(axis=1)(level4_1)

        level4_2 = Conv2D(256, (3, 3), activation='relu', padding='same', dilation_rate=(2, 2), kernel_initializer='he_normal')(level4_1)
        level4_2 = BatchNormalization(axis=1)(level4_2)

        level4_2 = Conv2D(256, (3, 3), activation='relu', padding='same', dilation_rate=(2, 2),
                          kernel_initializer='he_normal')(level4_2)
        level4_2 = BatchNormalization(axis=1)(level4_2)

        level4_2 = Conv2D(256, (3, 3), activation='relu', padding='same', dilation_rate=(2, 2),
                          kernel_initializer='he_normal')(level4_2)
        level4_2 = BatchNormalization(axis=1)(level4_2)

        # upsample

        level3_2 = Conv2DTranspose(128, (3, 3), activation='relu', padding='same', strides=(2, 2), kernel_initializer='he_normal')(level4_2)
        # level3_2 = BatchNormalization(axis=1)(level3_2)

        level3 = Average()([level3_1, level3_2])

        level2_2 = Conv2DTranspose(64, (3, 3), activation='relu', padding='same', strides=(2, 2), kernel_initializer='he_normal', name='level2_2')(level3)
        # level2_2 = BatchNormalization(axis=1)(level2_2)

        level2 = Average()([level2_1, level2_2])


        # Final conv
        Final_output = Conv2D(nb_q, (3, 3), name="final_conv2D", padding="same")(level2)

        color_output = UpSampling2D(size=(2, 2), dim_ordering="th", name='color_output')(Final_output)


        prior_factor = np.load("../../data/processed/ColorFusion_%s_prior_factor.npy" % 128)

        # sgd = SGD(lr=0.0005, decay=0.4e-5, momentum=0.9, nesterov=True)
        adam = Adam(lr=lr)  # 默认为0.001


        # Build model

        # 同时预测亮度信息和色度信息,即多任务输出
        # model = Model(input=[init], output=[color_output, gray_output], name=self.model_name)
        # model.compile(loss={'color_output': categorical_crossentropy_color(prior_factor), 'gray_output': 'mae'},
        #               loss_weights=[1., 1.], optimizer=adam)

        # 仅预测色度信息
        model = Model(input=[init], output=[color_output], name=self.model_name)
        model.compile(loss=categorical_crossentropy_color(prior_factor), optimizer=adam)

        self.model = model
        model.summary()

        plot_model(model, to_file='../../figures/%s_%s.png' % (self.model_name, sub_name), show_shapes=True, show_layer_names=True)
        return model

    def fit(self, **kwargs):
        return super(ResidualImageColorizationModel_trans, self).fit(**kwargs)

