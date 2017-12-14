from keras.callbacks import Callback
from theano import tensor as T # for NNET module
import keras.backend as K
import numpy as np
from keras.objectives import mean_absolute_error, mean_squared_error
''' Callbacks '''


class HistoryCheckpoint(Callback):
    '''Callback that records events
        into a `History` object.

        It then saves the history after each epoch into a file.
        To read the file into a python dict:
            history = {}
            with open(filename, "r") as f:
                history = eval(f.read())

        This may be unsafe since eval() will evaluate any string
        A safer alternative:

        import ast

        history = {}
        with open(filename, "r") as f:
            history = ast.literal_eval(f.read())

    '''

    def __init__(self, filename):
        super(Callback, self).__init__()
        self.filename = filename

    def on_train_begin(self, logs={}):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs={}):
        self.epoch.append(epoch)
        for k, v in logs.items():
            if k not in self.history:
                self.history[k] = []
            self.history[k].append(v)

        with open(self.filename, "w") as f:
            f.write(str(self.history))


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def loss_DSSIM_theano(y_true, y_pred):
    # expected net output is of shape [batch_size, row, col, image_channels]
    # e.g. [10, 480, 640, 3] for a batch of 10 640x480 RGB images
    # We need to shuffle this to [Batch_size, image_channels, row, col]
    # y_true = y_true.dimshuffle([0, 3, 1, 2])
    # y_pred = y_pred.dimshuffle([0, 3, 1, 2])

    # There are additional parameters for this function
    # Note: some of the 'modes' for edge behavior do not yet have a gradient definition in the Theano tree
    #   and cannot be used for learning
    patches_true = T.nnet.neighbours.images2neibs(y_true, [4, 4])
    patches_pred = T.nnet.neighbours.images2neibs(y_pred, [4, 4])

    u_true = K.mean(patches_true, axis=-1)
    u_pred = K.mean(patches_pred, axis=-1)
    var_true = K.var(patches_true, axis=-1)
    var_pred = K.var(patches_pred, axis=-1)
    std_true = K.sqrt(var_true + K.epsilon())
    std_pred = K.sqrt(var_pred + K.epsilon())
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)

    ssim /= K.clip(denom, K.epsilon(), np.inf)
    # ssim = tf.select(tf.is_nan(ssim), K.zeros_like(ssim), ssim)

    return K.mean((1.0 - ssim) / 2.0)


def mix_loss(y_true, y_pred):
    alpha = 0.3
    beta = 0
    gamma = 0.7
    return alpha * mean_absolute_error(y_true, y_pred) + beta * mean_squared_error(y_true, y_pred)\
           + gamma * loss_DSSIM_theano(y_true, y_pred)


def mix_loss_rmse(y_true, y_pred):
    alpha = 0
    beta = 0.3
    gamma = 0.7
    return alpha * mean_absolute_error(y_true, y_pred) + beta * root_mean_squared_error(y_true, y_pred)\
           + gamma * loss_DSSIM_theano(y_true, y_pred)




















