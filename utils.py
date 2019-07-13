import random
import keras


def r_activation():
    all_activation = ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu',
                      'tanh', 'sigmoid', 'hard_sigmoid', 'exponential', 'linear', None]
    activation_selected = random.sample(all_activation, 1)[0]
    return activation_selected


def r_initializer():
    all_initializer = ['Zeros', 'Ones', 'Constant', 'RandomNormal', 'RandomUniform',
                       'TruncatedNormal'
                       # 'VarianceScaling', 'Orthogonal', 'lecun_uniform', 'glorot_normal',
                       # 'glorot_uniform', 'he_normal', 'lecun_normal',
                       # 'Identity'
                       ]
    initializer_selected = random.sample(all_initializer, 1)[0]
    return initializer_selected


def r_regularizers():
    weight_decay = random.uniform(0.0, 0.01)
    all_regularizer = [keras.regularizers.l1(weight_decay), keras.regularizers.l2(weight_decay),
                       keras.regularizers.l1_l2(l1=weight_decay, l2=weight_decay), None]
    regularizer_selected = random.sample(all_regularizer, 1)[0]
    return regularizer_selected


def r_padding():
    all_padding = ['valid', 'causal', 'same']
    padding_selected = random.sample(all_padding, 1)[0]
    return padding_selected


def r_padding2():
    # all_padding = ['valid', 'same']
    all_padding = ['valid']

    padding_selected = random.sample(all_padding, 1)[0]
    return padding_selected


def r_interpolation():
    all_interpolation = ['nearest', 'bilinear']
    interpolation_selected = random.sample(all_interpolation, 1)[0]
    return interpolation_selected


def r_loss():
    all_loss = ['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error',
                'mean_squared_logarithmic_error', 'squared_hinge', 'hinge', 'categorical_hinge',
                'logcosh', 'categorical_crossentropy', 'sparse_categorical_crossentropy',
                'binary_crossentropy', 'kullback_leibler_divergence', 'poisson', 'cosine_proximity']
    loss_selected = random.sample(all_loss, 1)[0]
    return loss_selected


def r_optimizer():
    all_optimizer = ['sgd', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    optimizer_selected = random.sample(all_optimizer, 1)[0]
    return optimizer_selected
