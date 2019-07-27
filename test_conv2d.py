import numpy as np
import os
import importlib

from keras import backend as K
from keras.models import load_model
from keras.datasets import mnist, cifar10
from model_generation_new import RandomWeakModel
from conv2d_control_generator import RandomConv2dModel

from random_model import *


def set_keras_backend(backend='tensorflow'):
    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        importlib.reload(K)
        assert K.backend() == backend


def mnist_preprocess(x, backend='tensorflow'):
    x_test = np.copy(x)
    if backend in ['keras', 'tensorflow', 'cntk', 'theano']:
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    elif backend == 'mxnet':
        x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
    x_test = x_test.astype('float32')
    x_test /= 255
    return x_test


def cifar10_preprocess(x, backend='tensorflow'):
    x_test = x.copy()
    if backend in ['keras', 'tensorflow', 'cntk', 'theano']:
        x_test = x_test.astype('float32')
        mean = [125.307, 122.95, 113.865]
        std = [62.9932, 62.0887, 66.7048]
        for i in range(3):
            x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]
    elif backend == 'mxnet':
        x_test = x_test.transpose((0, 3, 1, 2))
        x_test = x_test.reshape(x_test.shape[0], 3, 32, 32)
        x_test = x_test.astype('float32')
        mean = [125.307, 122.95, 113.865]
        std = [62.9932, 62.0887, 66.7048]
        for i in range(3):
            x_test[:, i, :, :] = (x_test[:, i, :, :] - mean[i]) / std[i]
    return  x_test


def accuracy(y_pred, y):
    return np.sum(y_pred == y) / y.shape[0]


if __name__ == '__main__':
    (_, _), (x_test, y_test) = mnist.load_data()
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    y = y_test.reshape(y_test.shape[0])
    x = mnist_preprocess(x_test, backend='keras')

    set_keras_backend(backend='cntk')
    generator = RandomConv2dModel()
    # specify the config
    c_f = [0, 0, 0]

    # c_f[0] -> kernel_size
    #       0: odd
    #       1: even
    # c_f[1] -> strides
    #       0: 1
    #       1: odd
    #       2: even
    # c_f[2] -> padding
    #       0: same
    #       1: padding

    ll = generator.generate_layer((28, 28, 1), c_f)
    _loss, _op = generator.generate_compile()
    m_cntk, config_list, new_ll = generator.generate_model(ll, _loss, _op)

    m_cntk.save_weights('conv2d_model_weights.h5')
    save_str = str(c_f[0]) + str(c_f[1]) + str(c_f[2])
    m_cntk.save('conv2d_' + save_str +'.h5')
    print('Model saved!')
    print(config_list)

    y_cntk = m_cntk.predict(x, batch_size=200)
    y_cntk = np.argmax(y_cntk, axis=1)
    acc_cntk = accuracy(y_cntk, y)

    set_keras_backend()
    m_tf, tf_config_list, tf_ll = generator.generate_model(new_ll, _loss, _op, config_list)
    m_tf.load_weights('conv2d_model_weights.h5')
    y_tf = m_tf.predict(x, batch_size=200)
    y_tf = np.argmax(y_tf, axis=1)
    acc_tf = accuracy(y_tf, y)

    set_keras_backend(backend='theano')
    m_the, th_config_list, th_ll = generator.generate_model(new_ll, _loss, _op, config_list)
    m_the.load_weights('conv2d_model_weights.h5')
    y_the = m_the.predict(x, batch_size=200)
    y_the = np.argmax(y_the, axis=1)
    acc_the = accuracy(y_the, y)

    print('TF acc: {}'.format(acc_tf))
    print('CNTK acc: {}'.format(acc_cntk))
    print('Theano acc: {}'.format(acc_the))
