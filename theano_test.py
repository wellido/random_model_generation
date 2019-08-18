# coding: utf-8

# In[ ]:

import numpy as np

import theano.tensor as T
import theano
from keras.datasets import mnist
from keras.models import load_model, Model
import keras.backend as K
import os
import importlib


def set_keras_backend(backend='tensorflow'):
    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        importlib.reload(K)
        assert K.backend() == backend


def _postprocess_conv2d_output(conv_out, x,
                               padding, kernel_shape,
                               strides, data_format):
    if padding == 'same':
        if kernel_shape[2] % 2 == 0:
            i = (x.shape[2] + strides[0] - 1) // strides[0]
            conv_out = conv_out[:, :, :i, :]
        if kernel_shape[3] % 2 == 0:
            i = (x.shape[3] + strides[1] - 1) // strides[1]
            conv_out = conv_out[:, :, :, :i]
    if data_format == 'channels_last':
        conv_out = conv_out.dimshuffle((0, 2, 3, 1))
    return conv_out

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# # In[ ]:
x = T.matrix('x')
y = T.ivector('y')

set_keras_backend(backend='theano')
keras_model = load_model("models/DepthwiseConv2D_2.h5")

depthwise_kernel = keras_model.layers[0].depthwise_kernel
input_var = T.tensor4('inputs')
x = input_var.dimshuffle((0, 3, 1, 2))

depthwise_kernel = depthwise_kernel.dimshuffle((3, 2, 1, 0))
layer_1 = T.nnet.conv2d(x,
                        depthwise_kernel,
                        border_mode='half',
                        subsample=(1, 1),
                        filter_shape=(1, 1, 10, 10),
                        filter_dilation=(1, 1),
                        num_groups=1,

                        )
# print(layer_1.param)
conv_out = _postprocess_conv2d_output(layer_1, x, 'same',
                                      (1, 1, 10, 10), (1, 1), 'channels_last')

index = T.lscalar('index')
f = theano.function([input_var], conv_out, allow_input_downcast=True)
y_1 = f(x_test[0].reshape(1, 28, 28, 1))

new_keras_model = Model(inputs=keras_model.input, outputs=keras_model.layers[0].output)
new_keras_model.compile(loss=keras_model.loss,
                        optimizer=keras_model.optimizer,
                        metrics=['accuracy'])
y_2 = new_keras_model.predict(x_test[0].reshape(1, 28, 28, 1))
print("theano: ", np.sum(y_1))
print("keras: ", np.sum(y_2))
