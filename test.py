import numpy as np
import os
import importlib

from keras import backend as K
from keras.models import load_model
from keras.datasets import mnist, cifar10

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

def accuracy(y_pred,y):
    return np.sum(y_pred==y)/y.shape[0]


if __name__ == '__main__':
    generator = RandomModel()
    model = generator.generate_model((28, 28, 1))

    model.save('test_model.h5')
    print('Model saved!')

    (_,_),(x_test, y_test) = mnist.load_data()
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    y = y_test.reshape(y_test.shape[0])
    x = mnist_preprocess(x_test, backend='keras')
    x = x[:200]
    y = y[:200]

    # set_keras_backend(backend='cntk')
    # m_cntk = load_model('test_model.h5')
    # y_cntk = m_cntk.predict(x, batch_size=200)
    # y_cntk = np.argmax(y_cntk, axis=1)
    # acc_cntk = accuracy(y_cntk, y)

    # set_keras_backend()
    # m_tf = load_model('test_model.h5')
    y_tf = model.predict(x, batch_size=200)
    # y_tf = m_tf.predict(x, batch_size=200)
    y_tf = np.argmax(y_tf, axis=1)
    acc_tf = accuracy(y_tf, y)

    print('TF acc: {}'.format(acc_tf))
    # print('CNTK acc: {}'.format(acc_cntk))

    