import numpy as np
import os
import importlib

from keras import backend as K
from keras.datasets import mnist, cifar10
from one_layer_model_generator_order import Random1layerModel
from keras.models import load_model
import gc


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
    all_dict = {
        'Activation': 2,
        'Dropout': 3,
        'ActivityRegularization': 5,
        # 'Masking': 6,
        'LeakyReLU': 45,
        'PReLU': 46,
        'ELU': 47,
        'ThresholdedReLU': 48,
        'Softmax': 49,
        'ReLU': 50,
        'GaussianNoise': 51,
        'GaussianDropout': 52,
        'AlphaDropout': 53,
        'Dense': 1,
        'BatchNormalization': 7,

        'Flatten': 4,

        'SpatialDropout1D': 7,
        'Conv1D': 10,
        # 'SeparableConv1D': 12,
        'Cropping1D': 18,
        'UpSampling1D': 21,
        'ZeroPadding1D': 24,
        'MaxPooling1D': 27,
        'AveragePooling1D': 30,
        'GlobalMaxPooling1D': 33,
        'GlobalAveragePooling1D': 34,

        'SpatialDropout2D': 8,
        'Conv2D': 11,
        'SeparableConv2D': 13,
        'DepthwiseConv2D': 14,
        'Conv2DTranspose': 15,
        'Cropping2D': 19,
        'UpSampling2D': 22,
        'ZeroPadding2D': 25,
        'MaxPooling2D': 28,
        'AveragePooling2D': 31,
        'GlobalMaxPooling2D': 35,
        'GlobalAveragePooling2D': 36,
        'Conv2DControl': 58,

        # 'ConvLSTM2D': 44,
        'SpatialDropout3D': 9,
        'Conv3D': 16,
        'Conv3DTranspose': 17,
        'Cropping3D': 20,
        'UpSampling3D': 23,
        'ZeroPadding3D': 26,
        'MaxPooling3D': 29,
        'AveragePooling3D': 32,
        'GlobalMaxPooling3D': 37,
        'GlobalAveragePooling3D': 38,

        'SimpleRNN': 41,
        'GRU': 42,
        'LSTM': 43

    }

    dict_2d = {
        'Flatten': 4,
        'SpatialDropout2D': 8,
        'Conv2D': 11,
        'SeparableConv2D': 13,
        'DepthwiseConv2D': 14,
        'Conv2DTranspose': 15,
        'Cropping2D': 19,
        'ZeroPadding2D': 25,
        'MaxPooling2D': 28,
        'AveragePooling2D': 31,
        'GlobalMaxPooling2D': 35,
        'GlobalAveragePooling2D': 36,
        'Conv2DControl': 58,
        'Activation': 2,
        'Dropout': 3,
        'ActivityRegularization': 5,
        'LeakyReLU': 45,
        'PReLU': 46,
        'ELU': 47,
        'ThresholdedReLU': 48,
        'Softmax': 49,
        'ReLU': 50,
        'GaussianNoise': 51,
        'GaussianDropout': 52,
        'AlphaDropout': 53,
        'Dense': 1,
        'BatchNormalization': 7
    }
    dict_rnn = {
        'SimpleRNN': 41,
        'GRU': 42,
        'LSTM': 43
    }

    (_, _), (x_test, y_test) = mnist.load_data()
    x_test = x_test.reshape(x_test.shape[0], 28, 28)
    y = y_test.reshape(y_test.shape[0])
    x = mnist_preprocess(x_test, backend='keras')

    dict_2d_key = list(dict_2d.keys())
    dict_rnn_key = list(dict_rnn.keys())

    for layer_nam in dict_rnn:
        print(layer_nam)
        for i in range(1, 4):
            try:
                test_num = dict_rnn[layer_nam]

                set_keras_backend(backend='cntk')
                generator = Random1layerModel()
                ll = generator.generate_layer((28, 28), test_num)
                _loss, _op = generator.generate_compile()
                m_cntk, config_list, new_ll = generator.generate_model(ll, _loss, _op)
                m_cntk.save("one_layer_models/" + layer_nam + '_' + str(i) + '.h5')

                # cntk
                # y_cntk = m_cntk.predict(x, batch_size=200)
                # y_cntk = np.argmax(y_cntk, axis=1)
                # acc_cntk = accuracy(y_cntk, y)

                # tf
                # set_keras_backend()
                # m_tf = load_model("one_layer_models/" + layer_nam + '_' + str(i) + '.h5')
                # y_tf = m_tf.predict(x, batch_size=200)
                # y_tf = np.argmax(y_tf, axis=1)
                # acc_tf = accuracy(y_tf, y)

                # theano
                # set_keras_backend(backend='theano')
                # m_the = load_model("one_layer_models/" + layer_nam + '_' + str(i) + '.h5')
                # y_the = m_the.predict(x, batch_size=200)
                # y_the = np.argmax(y_the, axis=1)
                # acc_the = accuracy(y_the, y)
                #
                # print('TF acc: {}'.format(acc_tf))
                # print('CNTK acc: {}'.format(acc_cntk))
                # print('Theano acc: {}'.format(acc_the))
                del m_cntk
                # del m_tf
                # del m_the
                # gc.collect()
            except:
                print("skip one model")


# mmconvert -sf keras -iw one_layer_models/Conv2D_1.h5 -df pytorch -om one_layer_models_pytorch/Conv2D_1.pth