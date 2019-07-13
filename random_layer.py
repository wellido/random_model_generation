from utils import *
import random
import keras


class RandomLayers:
    def __init__(self):
        self.max_units = 50
        self.max_filter = 50
        self.max_cell = 50
        self.fix_dense = 10
        self.max_dropout_rate = 1.0
        self.max_regularization_factor = 0.01
        self.max_input_dim = 2000
        self.max_output_dim = 100
        self.layer_config = False
        self.units = 0
        self.batch_size = None
        self.input_shape = None
        self.is_first_layer = 0
        self.now_select_layer = 'all'

    def set_first_layer(self, is_first_layer):
        self.is_first_layer = is_first_layer

    def set_input_shape(self, input_shape):
        self.input_shape = input_shape

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    general_layer_map = {
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
        'Dense': 1
    }

    layer_map_0d = {
        'Flatten': 4
    }

    layer_map_1d = {
        'SpatialDropout1D': 7,
        'Conv1D': 10,
        # 'SeparableConv1D': 12,
        'Cropping1D': 18,
        'UpSampling1D': 21,
        'ZeroPadding1D': 24,
        'MaxPooling1D': 27,
        'AveragePooling1D': 30,
        'GlobalMaxPooling1D': 33,
        'GlobalAveragePooling1D': 34
        # 'LocallyConnected1D': 39,
    }

    layer_map_2d = {
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
        'GlobalAveragePooling2D': 36
        # 'LocallyConnected2D': 40,
    }

    layer_map_3d = {
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
    }

    layer_map_rnn = {
        'SimpleRNN': 41,
        'GRU': 42,
        'LSTM': 43,
    }

    layer_map_pooling = {
        'MaxPooling1D': 27,
        'AveragePooling1D': 30,
        'GlobalMaxPooling1D': 33,
        'GlobalAveragePooling1D': 34,
        'MaxPooling2D': 28,
        'AveragePooling2D': 31,
        'GlobalMaxPooling2D': 35,
        'GlobalAveragePooling2D': 36,
        'MaxPooling3D': 29,
        'AveragePooling3D': 32,
        'GlobalMaxPooling3D': 37,
        'GlobalAveragePooling3D': 38,
    }

    # Dense Layer
    def r_dense(self, _config=None):
        if self.layer_config:
            return keras.layers.Dense(_config[0],
                                      activation=_config[1],
                                      kernel_initializer=_config[2],
                                      bias_initializer=_config[3],
                                      kernel_regularizer=_config[4],
                                      bias_regularizer=_config[5],
                                      input_shape=_config[6],
                                      batch_size=_config[7])
        else:
            _activation = r_activation()
            _kernel_initializer = r_initializer()
            _bias_initializer = r_initializer()
            _kernel_regularizer = r_regularizers()
            _bias_regularizer = r_regularizers()
            _unit = random.randint(1, self.max_units)
            self.units = _unit
            return keras.layers.Dense(_unit,
                                      activation=_activation,
                                      kernel_initializer=_kernel_initializer,
                                      bias_initializer=_bias_initializer,
                                      kernel_regularizer=_kernel_regularizer,
                                      bias_regularizer=_bias_regularizer,
                                      input_shape=self.input_shape,
                                      batch_size=self.batch_size),\
                   [_unit, _activation, _kernel_initializer, _bias_initializer, _kernel_regularizer, _bias_regularizer, self.input_shape, self.batch_size]
        # if self.is_first_layer:
        #     if self.layer_config:
        #         return keras.layers.Dense.from_config(_config)
        #         # return keras.layers.Dense(_config[0],
        #         #                           activation=_config[1],
        #         #                           kernel_initializer=_config[2],
        #         #                           bias_initializer=_config[3],
        #         #                           kernel_regularizer=_config[4],
        #         #                           bias_regularizer=_config[5],
        #         #                           input_shape=self.input_shape,
        #         #                           batch_size=self.batch_size)
        #     else:
        #         return keras.layers.Dense(_unit,
        #                                   activation=_activation,
        #                                   kernel_initializer=_kernel_initializer,
        #                                   bias_initializer=_bias_initializer,
        #                                   kernel_regularizer=_kernel_regularizer,
        #                                   bias_regularizer=_bias_regularizer,
        #                                   input_shape=self.input_shape,
        #                                   batch_size=self.batch_size)
        # else:
        #     if self.layer_config:
        #         return keras.layers.Dense.from_config(_config)
        #     else:
        #         return keras.layers.Dense(_unit,
        #                                   activation=_activation,
        #                                   kernel_initializer=_kernel_initializer,
        #                                   bias_initializer=_bias_initializer,
        #                                   kernel_regularizer=_kernel_regularizer,
        #                                   bias_regularizer=_bias_regularizer,
        #                                   input_shape=None
        #                                   )

    # Dense Layer Without Activation
    def r_dense_without_activation(self, _unit, _config=None):
        if self.layer_config:
            return keras.layers.Dense(_config[0],
                                      activation=_config[1],
                                      kernel_initializer=_config[2],
                                      bias_initializer=_config[3],
                                      kernel_regularizer=_config[4],
                                      bias_regularizer=_config[5],
                                      input_shape=_config[6],
                                      batch_size=_config[7])
        else:
            _activation = None
            _kernel_initializer = r_initializer()
            _bias_initializer = r_initializer()
            _kernel_regularizer = r_regularizers()
            _bias_regularizer = r_regularizers()
            self.units = _unit
            return keras.layers.Dense(_unit,
                                      activation=_activation,
                                      kernel_initializer=_kernel_initializer,
                                      bias_initializer=_bias_initializer,
                                      kernel_regularizer=_kernel_regularizer,
                                      bias_regularizer=_bias_regularizer,
                                      input_shape=self.input_shape,
                                      batch_size=self.batch_size), \
                   [_unit, _activation, _kernel_initializer, _bias_initializer, _kernel_regularizer,
                    _bias_regularizer, self.input_shape, self.batch_size]
        # if self.is_first_layer:
        #     if self.layer_config:
        #         return keras.layers.Dense.from_config(_config)
        #     else:
        #         return keras.layers.Dense(_unit,
        #                                   activation=_activation,
        #                                   kernel_initializer=_kernel_initializer,
        #                                   bias_initializer=_bias_initializer,
        #                                   kernel_regularizer=_kernel_regularizer,
        #                                   bias_regularizer=_bias_regularizer,
        #                                   input_shape=self.input_shape,
        #                                   batch_size=self.batch_size)
        # else:
        #     if self.layer_config:
        #         return keras.layers.Dense.from_config(_config)
        #     else:
        #         return keras.layers.Dense(_unit,
        #                                   activation=_activation,
        #                                   kernel_initializer=_kernel_initializer,
        #                                   bias_initializer=_bias_initializer,
        #                                   kernel_regularizer=_kernel_regularizer,
        #                                   bias_regularizer=_bias_regularizer)

    # Activation Layer
    def r_activation(self, _config=None):
        if self.layer_config:
            return keras.layers.Activation(_config[0],
                                           input_shape=_config[1])
        else:
            _activation = r_activation()
            return keras.layers.Activation(_activation,
                                           input_shape=self.input_shape), \
                   [_activation, self.input_shape]
        # _activation = r_activation()
        # if self.is_first_layer:
        #     if self.layer_config:
        #         return keras.layers.Activation.from_config(_config)
        #     else:
        #         return keras.layers.Activation(_activation,
        #                                        input_shape=self.input_shape)
        # else:
        #     if self.layer_config:
        #         return keras.layers.Activation.from_config(_config)
        #     else:
        #         return keras.layers.Activation(_activation)

    # Dropout Layer
    def r_dropout(self, _config=None):

        if self.layer_config:
            return keras.layers.Dropout(_config[0],
                                        seed=_config[1],
                                        input_shape=_config[2],
                                        batch_size=_config[3])
        else:
            dropout_rate = random.uniform(0., self.max_dropout_rate)
            _seed = random.choice([None, 1, 2, 3, 4, 5])
            return keras.layers.Dropout(dropout_rate,
                                        seed=_seed,
                                        input_shape=self.input_shape,
                                        batch_size=self.batch_size), \
                   [dropout_rate, _seed, self.input_shape, self.batch_size]
        # if self.is_first_layer:
        #     if self.layer_config:
        #         return keras.layers.Dropout.from_config(_config)
        #     else:
        #         return keras.layers.Dropout(dropout_rate,
        #                                     seed=_seed,
        #                                     input_shape=self.input_shape,
        #                                     batch_size=self.batch_size)
        # else:
        #     if self.layer_config:
        #         return keras.layers.Dropout.from_config(_config)
        #     else:
        #         return keras.layers.Dropout(dropout_rate,
        #                                     seed=_seed)

    # Flatten Layer
    def r_flatten(self, _config=None):
        if self.layer_config:
            return keras.layers.Flatten(input_shape=_config[0],
                                        batch_size=_config[1])
        else:
            return keras.layers.Flatten(input_shape=self.input_shape,
                                        batch_size=self.batch_size), \
                   [self.input_shape, self.batch_size]
        # if self.is_first_layer:
        #     if self.layer_config:
        #         return keras.layers.Flatten.from_config(_config)
        #     else:
        #         return keras.layers.Flatten(input_shape=self.input_shape,
        #                                     batch_size=self.batch_size)
        # else:
        #     if self.layer_config:
        #         return keras.layers.Flatten.from_config(_config)
        #     else:
        #         return keras.layers.Flatten()

    # Input Layer
    def r_input(self, _config=None):
        if self.layer_config:
            return keras.layers.InputLayer(input_shape=_config[0],
                                           dtype=_config[1])
        else:
            return keras.layers.InputLayer(input_shape=self.input_shape,
                                           dtype='float32'), \
                   [self.input_shape, 'float32']

    # ActivityRegularization Layer
    def r_activity_regularization(self, _config=None):
        if self.layer_config:
            return keras.layers.ActivityRegularization(l1=_config[0],
                                                       l2=_config[1],
                                                       input_shape=_config[2],
                                                       batch_size=_config[3]
                                                       )
        else:
            l1_factor = random.uniform(0., self.max_regularization_factor)
            l2_factor = random.uniform(0., self.max_regularization_factor)
            return keras.layers.ActivityRegularization(l1=l1_factor,
                                                       l2=l2_factor,
                                                       input_shape=self.input_shape,
                                                       batch_size=self.batch_size), \
                   [l1_factor, l2_factor, self.input_shape, self.batch_size]


        # if self.is_first_layer:
        #     if self.layer_config:
        #         return keras.layers.ActivityRegularization.from_config(_config)
        #     else:
        #         return keras.layers.ActivityRegularization(l1=l1_factor,
        #                                                    l2=l2_factor,
        #                                                    input_shape=self.input_shape,
        #                                                    batch_size=self.batch_size)
        # else:
        #     if self.layer_config:
        #         return keras.layers.ActivityRegularization.from_config(_config)
        #     else:
        #         return keras.layers.ActivityRegularization(l1=l1_factor,
        #                                                    l2=l2_factor)

    # Masking Layer
    def r_masking(self, _config=None):
        if self.layer_config:
            return keras.layers.Masking(mask_value=_config[0],
                                        input_shape=_config[1],
                                        batch_size=_config[2])
        else:
            _mask_value = random.uniform(0., 1.0)
            return keras.layers.Masking(mask_value=_mask_value,
                                        input_shape=self.input_shape,
                                        batch_size=self.batch_size), \
                   [_mask_value, self.input_shape, self.batch_size]
        # if self.layer_config:
        #     return keras.layers.Masking.from_config(_config)
        # else:
        #     return keras.layers.Masking(mask_value=_mask_value,
        #                                 input_shape=self.input_shape,
        #                                 batch_size=self.batch_size)

    # SpatialDropout1D Layer
    def r_spatial_dropout1d(self, _config=None):
        if self.layer_config:
            return keras.layers.SpatialDropout1D(_config[0],
                                                 input_shape=_config[1],
                                                 batch_size=_config[2])
        else:
            dropout_rate = random.uniform(0., self.max_dropout_rate)
            return keras.layers.SpatialDropout1D(dropout_rate,
                                                 input_shape=self.input_shape,
                                                 batch_size=self.batch_size), \
                   [dropout_rate, self.input_shape, self.batch_size]
        # if self.is_first_layer:
        #     if self.layer_config:
        #         return keras.layers.SpatialDropout1D.from_config(_config)
        #     else:
        #         return keras.layers.SpatialDropout1D(dropout_rate,
        #                                              input_shape=self.input_shape,
        #                                              batch_size=self.batch_size)
        # else:
        #     if self.layer_config:
        #         return keras.layers.SpatialDropout1D.from_config(_config)
        #     else:
        #         return keras.layers.SpatialDropout1D(dropout_rate)

    # SpatialDropout2D Layer
    def r_spatial_dropout2d(self, _config=None):
        if self.layer_config:
            return keras.layers.SpatialDropout2D(_config[0],
                                                 input_shape=_config[1],
                                                 batch_size=_config[2])
        else:
            dropout_rate = random.uniform(0., self.max_dropout_rate)
            return keras.layers.SpatialDropout2D(dropout_rate,
                                                 input_shape=self.input_shape,
                                                 batch_size=self.batch_size), \
                   [dropout_rate, self.input_shape, self.batch_size]
        # if self.is_first_layer:
        #     if self.layer_config:
        #         return keras.layers.SpatialDropout2D.from_config(_config)
        #     else:
        #         return keras.layers.SpatialDropout2D(dropout_rate,
        #                                              input_shape=self.input_shape,
        #                                              batch_size=self.batch_size)
        # else:
        #     if self.layer_config:
        #         return keras.layers.SpatialDropout2D.from_config(_config)
        #     else:
        #         return keras.layers.SpatialDropout2D(dropout_rate)

    # SpatialDropout3D Layer
    def r_spatial_dropout3d(self, _config=None):
        if self.layer_config:
            return keras.layers.SpatialDropout3D(_config[0],
                                                 input_shape=_config[1],
                                                 batch_size=_config[2])
        else:
            dropout_rate = random.uniform(0., self.max_dropout_rate)
            return keras.layers.SpatialDropout3D(dropout_rate,
                                                 input_shape=self.input_shape,
                                                 batch_size=self.batch_size), \
                   [dropout_rate, self.input_shape, self.batch_size]
        # if self.is_first_layer:
        #     if self.layer_config:
        #         return keras.layers.SpatialDropout3D.from_config(_config)
        #     else:
        #         return keras.layers.SpatialDropout3D(dropout_rate,
        #                                              input_shape=self.input_shape,
        #                                              batch_size=self.batch_size)
        # else:
        #     if self.layer_config:
        #         return keras.layers.SpatialDropout3D.from_config(_config)
        #     else:
        #         return keras.layers.SpatialDropout3D(dropout_rate)

    # Conv1D Layer
    def r_convolution1d(self, _config=None):
        if self.layer_config:
            return keras.layers.Conv1D(_config[0],
                                       _config[1],
                                       strides=_config[3],
                                       padding=_config[4],
                                       activation=_config[5],
                                       kernel_initializer=_config[6],
                                       bias_initializer=_config[7],
                                       kernel_regularizer=_config[8],
                                       bias_regularizer=_config[9],
                                       activity_regularizer=_config[10],
                                       input_shape=_config[11],
                                       batch_size=_config[12])
        else:
            _filters = random.randint(1, self.max_filter)
            _kernel_size = random.randint(1, _filters)
            _strides = random.randint(1, _filters - _kernel_size + 1)
            _padding = r_padding()
            _activation = r_activation()
            _kernel_initializer = r_initializer()
            _bias_initializer = r_initializer()
            _kernel_regularizer = r_regularizers()
            _bias_regularizer = r_regularizers()
            _activity_regularizer = r_regularizers()
            return keras.layers.Conv1D(_filters,
                                       _kernel_size,
                                       strides=_strides,
                                       padding=_padding,
                                       activation=_activation,
                                       kernel_initializer=_kernel_initializer,
                                       bias_initializer=_bias_initializer,
                                       kernel_regularizer=_kernel_regularizer,
                                       bias_regularizer=_bias_regularizer,
                                       activity_regularizer=_activity_regularizer,
                                       input_shape=self.input_shape,
                                       batch_size=self.batch_size), \
                   [_filters, _kernel_size, _strides, _padding, _activation, _kernel_initializer, _bias_initializer,
                    _kernel_regularizer, _bias_regularizer, _activity_regularizer, self.input_shape, self.batch_size]
        # _filters = random.randint(1, self.max_filter)
        # _kernel_size = random.randint(1, _filters)
        # _strides = random.randint(1, _filters - _kernel_size + 1)
        # _padding = r_padding()
        # _activation = r_activation()
        # _kernel_initializer = r_initializer()
        # _bias_initializer = r_initializer()
        # _kernel_regularizer = r_regularizers()
        # _bias_regularizer = r_regularizers()
        # _activity_regularizer = r_regularizers()
        # if self.is_first_layer:
        #     if self.layer_config:
        #         return keras.layers.Conv1D.from_config(_config)
        #     else:
        #         return keras.layers.Conv1D(_filters,
        #                                    _kernel_size,
        #                                    strides=_strides,
        #                                    padding=_padding,
        #                                    activation=_activation,
        #                                    kernel_initializer=_kernel_initializer,
        #                                    bias_initializer=_bias_initializer,
        #                                    kernel_regularizer=_kernel_regularizer,
        #                                    bias_regularizer=_bias_regularizer,
        #                                    activity_regularizer=_activity_regularizer,
        #                                    input_shape=self.input_shape,
        #                                    batch_size=self.batch_size)
        # else:
        #     if self.layer_config:
        #         return keras.layers.Conv1D.from_config(_config)
        #     else:
        #         return keras.layers.Conv1D(_filters,
        #                                    _kernel_size,
        #                                    strides=_strides,
        #                                    padding=_padding,
        #                                    activation=_activation,
        #                                    kernel_initializer=_kernel_initializer,
        #                                    bias_initializer=_bias_initializer,
        #                                    kernel_regularizer=_kernel_regularizer,
        #                                    bias_regularizer=_bias_regularizer,
        #                                    activity_regularizer=_activity_regularizer)

    # Conv2D Layer
    def r_convolution2d(self, _config=None):
        if self.layer_config:
            return keras.layers.Conv2D(_config[0],
                                       _config[1],
                                       strides=_config[2],
                                       padding=_config[3],
                                       activation=_config[4],
                                       kernel_initializer=_config[5],
                                       bias_initializer=_config[6],
                                       kernel_regularizer=_config[7],
                                       bias_regularizer=_config[8],
                                       activity_regularizer=_config[9],
                                       input_shape=_config[10],
                                       batch_size=_config[11])
        else:
            _filters = random.randint(1, self.max_filter)
            _kernel_size = random.randint(1, _filters)
            _strides = random.randint(1, _filters - _kernel_size + 1)
            _padding = r_padding2()
            _activation = r_activation()
            _kernel_initializer = r_initializer()
            _bias_initializer = r_initializer()
            _kernel_regularizer = r_regularizers()
            _bias_regularizer = r_regularizers()
            _activity_regularizer = r_regularizers()
            return keras.layers.Conv2D(_filters,
                                       (_kernel_size, _kernel_size),
                                       strides=(_strides, _strides),
                                       padding=_padding,
                                       activation=_activation,
                                       kernel_initializer=_kernel_initializer,
                                       bias_initializer=_bias_initializer,
                                       kernel_regularizer=_kernel_regularizer,
                                       bias_regularizer=_bias_regularizer,
                                       activity_regularizer=_activity_regularizer,
                                       input_shape=self.input_shape,
                                       batch_size=self.batch_size), \
                   [_filters, (_kernel_size, _kernel_size), (_strides, _strides), _padding, _activation,
                    _kernel_initializer, _bias_initializer, _kernel_regularizer, _bias_regularizer, _activity_regularizer,
                    self.input_shape, self.batch_size]
        # if self.is_first_layer:
        #     if self.layer_config:
        #         return keras.layers.Conv2D.from_config(_config)
        #     else:
        #         return keras.layers.Conv2D(_filters,
        #                                    (_kernel_size, _kernel_size),
        #                                    strides=(_strides, _strides),
        #                                    padding=_padding,
        #                                    activation=_activation,
        #                                    kernel_initializer=_kernel_initializer,
        #                                    bias_initializer=_bias_initializer,
        #                                    kernel_regularizer=_kernel_regularizer,
        #                                    bias_regularizer=_bias_regularizer,
        #                                    activity_regularizer=_activity_regularizer,
        #                                    input_shape=self.input_shape,
        #                                    batch_size=self.batch_size)
        # else:
        #     if self.layer_config:
        #         return keras.layers.Conv2D.from_config(_config)
        #     else:
        #         return keras.layers.Conv2D(_filters,
        #                                    (_kernel_size, _kernel_size),
        #                                    strides=(_strides, _strides),
        #                                    padding=_padding,
        #                                    activation=_activation,
        #                                    kernel_initializer=_kernel_initializer,
        #                                    bias_initializer=_bias_initializer,
        #                                    kernel_regularizer=_kernel_regularizer,
        #                                    bias_regularizer=_bias_regularizer,
        #                                    activity_regularizer=_activity_regularizer)

    # SeparableConv1D Layer
    def r_separable_conv1d(self, _config=None):
        if self.layer_config:
            return keras.layers.SeparableConv1D(_config[0],
                                                _config[1],
                                                _config[2],
                                                padding=_config[3],
                                                activation=_config[4],
                                                kernel_initializer=_config[5],
                                                bias_initializer=_config[6],
                                                kernel_regularizer=_config[7],
                                                bias_regularizer=_config[8],
                                                activity_regularizer=_config[9],
                                                input_shape=_config[10],
                                                batch_size=_config[11])
        else:
            _filters = random.randint(1, self.max_filter)
            _kernel_size = random.randint(1, _filters)
            _strides = random.randint(1, _filters - _kernel_size + 1)
            _padding = r_padding2()
            _activation = r_activation()
            _kernel_initializer = r_initializer()
            _bias_initializer = r_initializer()
            _kernel_regularizer = r_regularizers()
            _bias_regularizer = r_regularizers()
            _activity_regularizer = r_regularizers()
            return keras.layers.SeparableConv1D(_filters,
                                                _kernel_size,
                                                _strides,
                                                padding=_padding,
                                                activation=_activation,
                                                kernel_initializer=_kernel_initializer,
                                                bias_initializer=_bias_initializer,
                                                kernel_regularizer=_kernel_regularizer,
                                                bias_regularizer=_bias_regularizer,
                                                activity_regularizer=_activity_regularizer,
                                                input_shape=self.input_shape,
                                                batch_size=self.batch_size), \
        [_filters, _kernel_size, _strides, _padding, _activation, _kernel_initializer, _bias_initializer,
         _kernel_regularizer, _bias_initializer, _kernel_regularizer, _bias_regularizer, _activity_regularizer,
         self.input_shape, self.batch_size]
        # if self.is_first_layer:
        #     if self.layer_config:
        #         return keras.layers.SeparableConv1D.from_config(_config)
        #     else:
        #         return keras.layers.SeparableConv1D(_filters,
        #                                             _kernel_size,
        #                                             _strides,
        #                                             padding=_padding,
        #                                             activation=_activation,
        #                                             kernel_initializer=_kernel_initializer,
        #                                             bias_initializer=_bias_initializer,
        #                                             kernel_regularizer=_kernel_regularizer,
        #                                             bias_regularizer=_bias_regularizer,
        #                                             activity_regularizer=_activity_regularizer,
        #                                             input_shape=self.input_shape,
        #                                             batch_size=self.batch_size)
        # else:
        #     if self.layer_config:
        #         return keras.layers.SeparableConv1D.from_config(_config)
        #     else:
        #         return keras.layers.SeparableConv1D(_filters,
        #                                             _kernel_size,
        #                                             _strides,
        #                                             padding=_padding,
        #                                             activation=_activation,
        #                                             kernel_initializer=_kernel_initializer,
        #                                             bias_initializer=_bias_initializer,
        #                                             kernel_regularizer=_kernel_regularizer,
        #                                             bias_regularizer=_bias_regularizer,
        #                                             activity_regularizer=_activity_regularizer)

    # SeparableConv2D Layer
    def r_separable_conv2d(self, _config=None):
        if self.layer_config:
            return keras.layers.SeparableConv2D(_config[0],
                                                _config[1],
                                                strides=_config[2],
                                                padding=_config[3],
                                                activation=_config[4],
                                                kernel_initializer=_config[5],
                                                bias_initializer=_config[6],
                                                kernel_regularizer=_config[7],
                                                bias_regularizer=_config[8],
                                                activity_regularizer=_config[9],
                                                input_shape=_config[10],
                                                batch_size=_config[11])
        else:
            _filters = random.randint(1, self.max_filter)
            _kernel_size = random.randint(1, _filters)
            _strides = random.randint(1, _filters - _kernel_size + 1)
            _padding = r_padding2()
            _activation = r_activation()
            _kernel_initializer = r_initializer()
            _bias_initializer = r_initializer()
            _kernel_regularizer = r_regularizers()
            _bias_regularizer = r_regularizers()
            _activity_regularizer = r_regularizers()
            return keras.layers.SeparableConv2D(_filters,
                                                (_kernel_size, _kernel_size),
                                                strides=(_strides, _strides),
                                                padding=_padding,
                                                activation=_activation,
                                                kernel_initializer=_kernel_initializer,
                                                bias_initializer=_bias_initializer,
                                                kernel_regularizer=_kernel_regularizer,
                                                bias_regularizer=_bias_regularizer,
                                                activity_regularizer=_activity_regularizer,
                                                input_shape=self.input_shape,
                                                batch_size=self.batch_size), \
                   [_filters, (_kernel_size, _kernel_size), (_strides, _strides), _padding, _activation, _kernel_initializer,
                    _bias_initializer, _kernel_regularizer, _bias_regularizer, _activity_regularizer, self.input_shape,
                    self.batch_size]
        # if self.is_first_layer:
        #     if self.layer_config:
        #         return keras.layers.SeparableConv2D.from_config(_config)
        #     else:
        #         return keras.layers.SeparableConv2D(_filters,
        #                                             (_kernel_size, _kernel_size),
        #                                             strides=(_strides, _strides),
        #                                             padding=_padding,
        #                                             activation=_activation,
        #                                             kernel_initializer=_kernel_initializer,
        #                                             bias_initializer=_bias_initializer,
        #                                             kernel_regularizer=_kernel_regularizer,
        #                                             bias_regularizer=_bias_regularizer,
        #                                             activity_regularizer=_activity_regularizer,
        #                                             input_shape=self.input_shape,
        #                                             batch_size=self.batch_size)
        # else:
        #     if self.layer_config:
        #         return keras.layers.SeparableConv2D.from_config(_config)
        #     else:
        #         return keras.layers.SeparableConv2D(_filters,
        #                                             (_kernel_size, _kernel_size),
        #                                             strides=(_strides, _strides),
        #                                             padding=_padding,
        #                                             activation=_activation,
        #                                             kernel_initializer=_kernel_initializer,
        #                                             bias_initializer=_bias_initializer,
        #                                             kernel_regularizer=_kernel_regularizer,
        #                                             bias_regularizer=_bias_regularizer,
        #                                             activity_regularizer=_activity_regularizer)

    # DepthwiseConv2D Layer
    def r_depthwise_conv2d(self, _config=None):
        if self.layer_config:
            return keras.layers.DepthwiseConv2D(_config[0],
                                                strides=_config[1],
                                                padding=_config[2],
                                                activation=_config[3],
                                                kernel_initializer=_config[4],
                                                bias_initializer=_config[5],
                                                kernel_regularizer=_config[6],
                                                bias_regularizer=_config[7],
                                                activity_regularizer=_config[8],
                                                input_shape=_config[9],
                                                batch_size=_config[10])
        else:
            _kernel_size = random.randint(1, self.max_filter)
            _strides = random.randint(1, self.max_filter - _kernel_size + 1)
            _padding = r_padding2()
            _activation = r_activation()
            _kernel_initializer = r_initializer()
            _bias_initializer = r_initializer()
            _kernel_regularizer = r_regularizers()
            _bias_regularizer = r_regularizers()
            _activity_regularizer = r_regularizers()
            return keras.layers.DepthwiseConv2D(_kernel_size,
                                                strides=(_strides, _strides),
                                                padding=_padding,
                                                activation=_activation,
                                                kernel_initializer=_kernel_initializer,
                                                bias_initializer=_bias_initializer,
                                                kernel_regularizer=_kernel_regularizer,
                                                bias_regularizer=_bias_regularizer,
                                                activity_regularizer=_activity_regularizer,
                                                input_shape=self.input_shape,
                                                batch_size=self.batch_size), \
                   [_kernel_size, (_strides, _strides), _padding, _activation, _kernel_initializer, _bias_initializer,
                    _kernel_regularizer, _bias_regularizer, _activity_regularizer, self.input_shape, self.batch_size]
        # if self.is_first_layer:
        #     if self.layer_config:
        #         return keras.layers.DepthwiseConv2D.from_config(_config)
        #     else:
        #         return keras.layers.DepthwiseConv2D(_kernel_size,
        #                                             strides=(_strides, _strides),
        #                                             padding=_padding,
        #                                             activation=_activation,
        #                                             kernel_initializer=_kernel_initializer,
        #                                             bias_initializer=_bias_initializer,
        #                                             kernel_regularizer=_kernel_regularizer,
        #                                             bias_regularizer=_bias_regularizer,
        #                                             activity_regularizer=_activity_regularizer,
        #                                             input_shape=self.input_shape,
        #                                             batch_size=self.batch_size)
        # else:
        #     if self.layer_config:
        #         return keras.layers.DepthwiseConv2D.from_config(_config)
        #     else:
        #         return keras.layers.DepthwiseConv2D(_kernel_size,
        #                                             strides=(_strides, _strides),
        #                                             padding=_padding,
        #                                             activation=_activation,
        #                                             kernel_initializer=_kernel_initializer,
        #                                             bias_initializer=_bias_initializer,
        #                                             kernel_regularizer=_kernel_regularizer,
        #                                             bias_regularizer=_bias_regularizer,
        #                                             activity_regularizer=_activity_regularizer)

    # Conv2DTranspose Layer
    def r_conv2d_transpose(self, _config=None):
        if self.layer_config:
            return keras.layers.Conv2DTranspose(_config[0],
                                                _config[1],
                                                strides=_config[2],
                                                padding=_config[3],
                                                activation=_config[4],
                                                kernel_initializer=_config[5],
                                                bias_initializer=_config[6],
                                                kernel_regularizer=_config[7],
                                                bias_regularizer=_config[8],
                                                activity_regularizer=_config[9],
                                                input_shape=_config[10],
                                                batch_size=_config[11]
                                                )
        else:
            _filters = random.randint(1, self.max_filter)
            _kernel_size = random.randint(1, _filters)
            _strides = random.randint(1, _filters - _kernel_size + 1)
            _padding = r_padding2()
            _activation = r_activation()
            _kernel_initializer = r_initializer()
            _bias_initializer = r_initializer()
            _kernel_regularizer = r_regularizers()
            _bias_regularizer = r_regularizers()
            _activity_regularizer = r_regularizers()
            return keras.layers.Conv2DTranspose(_filters,
                                                (_kernel_size, _kernel_size),
                                                strides=(_strides, _strides),
                                                padding=_padding,
                                                activation=_activation,
                                                kernel_initializer='glorot_uniform',
                                                bias_initializer='zeros',
                                                kernel_regularizer=_kernel_regularizer,
                                                bias_regularizer=_bias_regularizer,
                                                activity_regularizer=_activity_regularizer,
                                                input_shape=self.input_shape,
                                                batch_size=self.batch_size
                                                ), \
                   [_filters, (_kernel_size, _kernel_size), (_strides, _strides), _padding, _activation, 'glorot_uniform',
                    'zeros', _kernel_regularizer, _bias_regularizer, _activity_regularizer, self.input_shape,
                    self.batch_size]
        # if self.is_first_layer:
        #     if self.layer_config:
        #         return keras.layers.Conv2DTranspose.from_config(_config)
        #     else:
        #         return keras.layers.Conv2DTranspose(_filters,
        #                                             (_kernel_size, _kernel_size),
        #                                             strides=(_strides, _strides),
        #                                             padding=_padding,
        #                                             activation=_activation,
        #                                             kernel_initializer='glorot_uniform',
        #                                             bias_initializer='zeros',
        #                                             kernel_regularizer=_kernel_regularizer,
        #                                             bias_regularizer=_bias_regularizer,
        #                                             activity_regularizer=_activity_regularizer,
        #                                             input_shape=self.input_shape,
        #                                             batch_size=self.batch_size
        #                                             )
        # else:
        #     return keras.layers.Conv2DTranspose(_filters,
        #                                         (_kernel_size, _kernel_size),
        #                                         strides=(_strides, _strides),
        #                                         padding=_padding,
        #                                         activation=_activation,
        #                                         kernel_initializer='glorot_uniform',
        #                                         bias_initializer='zeros',
        #                                         kernel_regularizer=_kernel_regularizer,
        #                                         bias_regularizer=_bias_regularizer,
        #                                         activity_regularizer=_activity_regularizer
        #                                         )

    # Conv3D Layer
    def r_convolution3d(self, _config=None):
        if self.layer_config:
            return keras.layers.Conv3D(_config[0],
                                       _config[1],
                                       strides=_config[2],
                                       padding=_config[3],
                                       activation=_config[4],
                                       kernel_initializer=_config[5],
                                       bias_initializer=_config[6],
                                       kernel_regularizer=_config[7],
                                       bias_regularizer=_config[8],
                                       activity_regularizer=_config[9],
                                       input_shape=_config[10],
                                       batch_size=_config[11])
        else:

            _filters = random.randint(1, self.max_filter)
            _kernel_size = random.randint(1, _filters)
            _strides = random.randint(1, _filters - _kernel_size + 1)
            _padding = r_padding2()
            _activation = r_activation()
            _kernel_initializer = r_initializer()
            _bias_initializer = r_initializer()
            _kernel_regularizer = r_regularizers()
            _bias_regularizer = r_regularizers()
            _activity_regularizer = r_regularizers()
            return keras.layers.Conv3D(_filters,
                                       (_kernel_size, _kernel_size, _kernel_size),
                                       strides=(_strides, _strides, _strides),
                                       padding=_padding,
                                       activation=_activation,
                                       kernel_initializer=_kernel_initializer,
                                       bias_initializer=_bias_initializer,
                                       kernel_regularizer=_kernel_regularizer,
                                       bias_regularizer=_bias_regularizer,
                                       activity_regularizer=_activity_regularizer,
                                       input_shape=self.input_shape,
                                       batch_size=self.batch_size), \
                   [_filters, (_kernel_size, _kernel_size, _kernel_size), _padding, _activation, _kernel_initializer,
                    _bias_initializer, _kernel_regularizer, _bias_regularizer, _activity_regularizer, self.input_shape,
                    self.batch_size]
        # if self.is_first_layer:
        #     if self.layer_config:
        #         return keras.layers.Conv3D.from_config(_config)
        #     else:
        #         return keras.layers.Conv3D(_filters,
        #                                    (_kernel_size, _kernel_size, _kernel_size),
        #                                    strides=(_strides, _strides, _strides),
        #                                    padding=_padding,
        #                                    activation=_activation,
        #                                    kernel_initializer=_kernel_initializer,
        #                                    bias_initializer=_bias_initializer,
        #                                    kernel_regularizer=_kernel_regularizer,
        #                                    bias_regularizer=_bias_regularizer,
        #                                    activity_regularizer=_activity_regularizer,
        #                                    input_shape=self.input_shape,
        #                                    batch_size=self.batch_size)
        # else:
        #     if self.layer_config:
        #         return keras.layers.Conv3D.from_config(_config)
        #     else:
        #         return keras.layers.Conv3D(_filters,
        #                                    (_kernel_size, _kernel_size, _kernel_size),
        #                                    strides=(_strides, _strides, _strides),
        #                                    padding=_padding,
        #                                    activation=_activation,
        #                                    kernel_initializer=_kernel_initializer,
        #                                    bias_initializer=_bias_initializer,
        #                                    kernel_regularizer=_kernel_regularizer,
        #                                    bias_regularizer=_bias_regularizer,
        #                                    activity_regularizer=_activity_regularizer)

    # Conv3DTranspose Layer
    def r_conv3d_transpose(self, _config=None):
        if self.layer_config:
            return keras.layers.Conv3DTranspose(_config[0],
                                                _config[1],
                                                strides=_config[2],
                                                padding=_config[3],
                                                activation=_config[4],
                                                kernel_initializer=_config[5],
                                                bias_initializer=_config[6],
                                                kernel_regularizer=_config[7],
                                                bias_regularizer=_config[8],
                                                activity_regularizer=_config[9],
                                                input_shape=_config[10],
                                                batch_size=_config[11]
                                                )
        else:

            _filters = random.randint(1, self.max_filter)
            _kernel_size = random.randint(1, _filters)
            _strides = random.randint(1, _filters - _kernel_size + 1)
            _padding = r_padding2()
            _activation = r_activation()
            _kernel_initializer = r_initializer()
            _bias_initializer = r_initializer()
            _kernel_regularizer = r_regularizers()
            _bias_regularizer = r_regularizers()
            _activity_regularizer = r_regularizers()
            return keras.layers.Conv3DTranspose(_filters,
                                                (_kernel_size, _kernel_size, _kernel_size),
                                                strides=(_strides, _strides, _strides),
                                                padding=_padding,
                                                activation=_activation,
                                                kernel_initializer=_kernel_initializer,
                                                bias_initializer=_bias_initializer,
                                                kernel_regularizer=_kernel_regularizer,
                                                bias_regularizer=_bias_regularizer,
                                                activity_regularizer=_activity_regularizer,
                                                input_shape=self.input_shape,
                                                batch_size=self.batch_size
                                                ), \
                   [_filters, (_kernel_size, _kernel_size, _kernel_size), _padding, _activation, _kernel_initializer,
                    _bias_initializer, _kernel_regularizer, _bias_regularizer, _activity_regularizer, self.input_shape,
                    self.batch_size]
        # if self.is_first_layer:
        #     if self.layer_config:
        #         return keras.layers.Conv3DTranspose.from_config(_config)
        #     else:
        #         return keras.layers.Conv3DTranspose(_filters,
        #                                             (_kernel_size, _kernel_size, _kernel_size),
        #                                             strides=(_strides, _strides, _strides),
        #                                             padding=_padding, activation=_activation,
        #                                             kernel_initializer=_kernel_initializer,
        #                                             bias_initializer=_bias_initializer,
        #                                             kernel_regularizer=_kernel_regularizer,
        #                                             bias_regularizer=_bias_regularizer,
        #                                             activity_regularizer=_activity_regularizer,
        #                                             input_shape=self.input_shape,
        #                                             batch_size=self.batch_size
        #                                             )
        # else:
        #     if self.layer_config:
        #         return keras.layers.Conv3DTranspose.from_config(_config)
        #     else:
        #         return keras.layers.Conv3DTranspose(_filters,
        #                                             (_kernel_size, _kernel_size, _kernel_size),
        #                                             strides=(_strides, _strides, _strides),
        #                                             padding=_padding, activation=_activation,
        #                                             kernel_initializer=_kernel_initializer,
        #                                             bias_initializer=_bias_initializer,
        #                                             kernel_regularizer=_kernel_regularizer,
        #                                             bias_regularizer=_bias_regularizer,
        #                                             activity_regularizer=_activity_regularizer,
        #                                             )

    # Cropping1D Layer
    def r_cropping1d(self, _config=None):
        if self.layer_config:
            return keras.layers.Cropping1D(cropping=_config[0],
                                           input_shape=_config[1],
                                           batch_size=_config[2])
        else:
            _cropping = random.randint(0, 2)
            return keras.layers.Cropping1D(cropping=_cropping,
                                           input_shape=self.input_shape,
                                           batch_size=self.batch_size), \
                   [_cropping, self.input_shape, self.batch_size]

        # if self.is_first_layer:
        #     if self.layer_config:
        #         return keras.layers.Cropping1D.from_config(_config)
        #     else:
        #         return keras.layers.Cropping1D(cropping=_cropping,
        #                                        input_shape=self.input_shape,
        #                                        batch_size=self.batch_size)
        # else:
        #     if self.layer_config:
        #         return keras.layers.Cropping1D.from_config(_config)
        #     else:
        #         return keras.layers.Cropping1D(cropping=_cropping)

    # Cropping2D Layer
    def r_cropping2d(self, _config=None):
        if self.layer_config:
            return keras.layers.Cropping2D(cropping=_config[0],
                                           input_shape=_config[1],
                                           batch_size=_config[2])
        else:
            _cropping = random.randint(0, 2)
            return keras.layers.Cropping2D(cropping=_cropping,
                                           input_shape=self.input_shape,
                                           batch_size=self.batch_size), \
                   [_cropping, self.input_shape, self.batch_size]
        # if self.is_first_layer:
        #     if self.layer_config:
        #         return keras.layers.Cropping2D.from_config(_config)
        #     else:
        #         return keras.layers.Cropping2D(cropping=_cropping,
        #                                        input_shape=self.input_shape,
        #                                        batch_size=self.batch_size)
        # else:
        #     if self.layer_config:
        #         return keras.layers.Cropping2D.from_config(_config)
        #     else:
        #         return keras.layers.Cropping2D(cropping=_cropping)

    # Cropping3D Layer
    def r_cropping3d(self, _config=None):
        if self.layer_config:
            return keras.layers.Cropping3D(cropping=_config[0],
                                           input_shape=_config[1],
                                           batch_size=_config[2])
        else:
            _cropping = random.randint(0, 2)
            return keras.layers.Cropping3D(cropping=_cropping,
                                           input_shape=self.input_shape,
                                           batch_size=self.batch_size), \
                   [_cropping, self.input_shape, self.batch_size]
        # _cropping = random.randint(0, 2)
        # if self.is_first_layer:
        #     if self.layer_config:
        #         return keras.layers.Cropping3D.from_config(_config)
        #     else:
        #         return keras.layers.Cropping3D(cropping=_cropping,
        #                                        input_shape=self.input_shape,
        #                                        batch_size=self.batch_size)
        # else:
        #     if self.layer_config:
        #         return keras.layers.Cropping3D.from_config(_config)
        #     else:
        #         return keras.layers.Cropping3D(cropping=_cropping)

    # UpSampling1D Layer
    def r_up_sampling1d(self, _config=None):
        if self.layer_config:
            return keras.layers.UpSampling1D(size=_config[0],
                                             input_shape=_config[1],
                                             batch_size=_config[2])
        else:

            _size = random.randint(0, 2)
            return keras.layers.UpSampling1D(size=_size,
                                             input_shape=self.input_shape,
                                             batch_size=self.batch_size), \
                   [_size, self.input_shape, self.batch_size]
        # if self.is_first_layer:
        #     if self.layer_config:
        #         return keras.layers.UpSampling1D.from_config(_config)
        #     else:
        #         return keras.layers.UpSampling1D(size=_size,
        #                                          input_shape=self.input_shape,
        #                                          batch_size=self.batch_size)
        # else:
        #     if self.layer_config:
        #         return keras.layers.UpSampling1D.from_config(_config)
        #     else:
        #         return keras.layers.UpSampling1D(size=_size)

    # UpSampling2D Layer
    def r_up_sampling2d(self, _config=None):
        if self.layer_config:
            return keras.layers.UpSampling2D(size=_config[0],
                                             interpolation=_config[1],
                                             input_shape=_config[2],
                                             batch_size=_config[3])
        else:
            _size = random.randint(0, 2)
            _interpolation = r_interpolation()
            return keras.layers.UpSampling2D(size=_size,
                                             interpolation=_interpolation,
                                             input_shape=self.input_shape,
                                             batch_size=self.batch_size), \
                   [_size, _interpolation, self.input_shape, self.batch_size]
        # if self.is_first_layer:
        #     if self.layer_config:
        #         return keras.layers.UpSampling2D.from_config(_config)
        #     else:
        #         return keras.layers.UpSampling2D(size=_size,
        #                                          interpolation=_interpolation,
        #                                          input_shape=self.input_shape,
        #                                          batch_size=self.batch_size)
        # else:
        #     if self.layer_config:
        #         return keras.layers.UpSampling2D.from_config(_config)
        #     else:
        #         return keras.layers.UpSampling2D(size=_size,
        #                                          interpolation=_interpolation)

    # UpSampling3D Layer
    def r_up_sampling3d(self, _config=None):
        if self.layer_config:
            return keras.layers.UpSampling3D(size=_config[0],
                                             input_shape=_config[1],
                                             batch_size=_config[2])
        else:
            _size = random.randint(0, 2)
            return keras.layers.UpSampling3D(size=_size,
                                             input_shape=self.input_shape,
                                             batch_size=self.batch_size), \
                   [_size, self.input_shape, self.batch_size]
        # if self.is_first_layer:
        #     if self.layer_config:
        #         return keras.layers.UpSampling3D.from_config(_config)
        #     else:
        #         return keras.layers.UpSampling3D(size=_size,
        #                                          input_shape=self.input_shape,
        #                                          batch_size=self.batch_size)
        # else:
        #     if self.layer_config:
        #         return keras.layers.UpSampling3D.from_config(_config)
        #     else:
        #         return keras.layers.UpSampling3D(size=_size)

    # ZeroPadding1D Layer
    def r_zero_padding1d(self, _config=None):
        if self.layer_config:
            return keras.layers.ZeroPadding1D(padding=_config[0],
                                              input_shape=_config[1],
                                              batch_size=_config[2])
        else:
            _choose = random.randint(0, 1)
            _padding = random.randint(0, 9)
            return keras.layers.ZeroPadding1D(padding=(_padding, _padding),
                                              input_shape=self.input_shape,
                                              batch_size=self.batch_size), \
                   [(_padding, _padding), self.input_shape, self.batch_size]
        # if self.layer_config:
        #     return keras.layers.ZeroPadding1D.from_config(_config)
        # if _choose:
        #     if self.is_first_layer:
        #         return keras.layers.ZeroPadding1D(padding=(_padding, _padding),
        #                                           input_shape=self.input_shape,
        #                                           batch_size=self.batch_size)
        #
        #     else:
        #         return keras.layers.ZeroPadding1D(padding=(_padding, _padding))
        # else:
        #     return keras.layers.ZeroPadding1D(padding=_padding)

    # ZeroPadding2D Layer
    def r_zero_padding2d(self, _config=None):
        if self.layer_config:
            return keras.layers.ZeroPadding2D(padding=_config[0],
                                              data_format=_config[1],
                                              input_shape=_config[2],
                                              batch_size=_config[3])
        else:
            _padding = random.randint(0, 9)
            return keras.layers.ZeroPadding2D(padding=_padding,
                                              data_format=None,
                                              input_shape=self.input_shape,
                                              batch_size=self.batch_size), \
                   [_padding, None, self.input_shape, self.batch_size]
        # if self.layer_config:
        #     return keras.layers.ZeroPadding2D.from_config(_config)
        # _padding = random.randint(0, 9)
        # if self.is_first_layer:
        #     return keras.layers.ZeroPadding2D(padding=_padding,
        #                                       data_format=None,
        #                                       input_shape=self.input_shape,
        #                                       batch_size=self.batch_size)
        # else:
        #     return keras.layers.ZeroPadding2D(padding=_padding,
        #                                       data_format=None)

    # ZeroPadding3D Layer
    def r_zero_padding3d(self, _config=None):
        if self.layer_config:
            return keras.layers.ZeroPadding3D(padding=_config[0],
                                              data_format=_config[1],
                                              input_shape=_config[2],
                                              batch_size=_config[3])
        else:
            _padding = random.randint(0, 9)
            return keras.layers.ZeroPadding3D(padding=_padding,
                                              data_format=None,
                                              input_shape=self.input_shape,
                                              batch_size=self.batch_size), \
                   [_padding, None, self.input_shape, self.batch_size]
        # if self.layer_config:
        #     return keras.layers.ZeroPadding3D.from_config(_config)
        # _padding = random.randint(0, 9)
        # if self.is_first_layer:
        #     return keras.layers.ZeroPadding3D(padding=_padding,
        #                                       data_format=None,
        #                                       input_shape=self.input_shape,
        #                                       batch_size=self.batch_size)
        # else:
        #     return keras.layers.ZeroPadding3D(padding=_padding,
        #                                       data_format=None)

    # MaxPooling1D Layer
    def r_max_pooling1d(self, _config=None):
        if self.layer_config:
            return keras.layers.MaxPooling1D(pool_size=_config[0],
                                             padding=_config[1],
                                             input_shape=_config[2],
                                             batch_size=_config[3])
        else:
            _pool_size = random.randint(1, 3)
            _padding = r_padding2()
            return keras.layers.MaxPooling1D(pool_size=_pool_size,
                                             padding=_padding,
                                             input_shape=self.input_shape,
                                             batch_size=self.batch_size), \
                   [_pool_size, _padding, self.input_shape, self.batch_size]
        # if self.layer_config:
        #     return keras.layers.MaxPooling1D.from_config(_config)
        # _pool_size = random.randint(1, 3)
        # _padding = r_padding2()
        # if self.is_first_layer:
        #     return keras.layers.MaxPooling1D(pool_size=_pool_size,
        #                                      padding=_padding,
        #                                      input_shape=self.input_shape,
        #                                      batch_size=self.batch_size)
        # else:
        #     return keras.layers.MaxPooling1D(pool_size=_pool_size,
        #                                      padding=_padding)

    # MaxPooling2D Layer
    def r_max_pooling2d(self, _config=None):
        if self.layer_config:
            return keras.layers.MaxPooling2D(pool_size=_config[0],
                                             padding=_config[1],
                                             input_shape=_config[2],
                                             batch_size=_config[3])
        else:
            _pool_size = random.randint(1, 3)
            _padding = r_padding2()
            return keras.layers.MaxPooling2D(pool_size=_pool_size,
                                             padding=_padding,
                                             input_shape=self.input_shape,
                                             batch_size=self.batch_size), \
                   [_pool_size, _padding, self.input_shape, self.batch_size]
        # if self.layer_config:
        #     return keras.layers.MaxPooling2D.from_config(_config)
        # _pool_size = random.randint(1, 3)
        # _padding = r_padding2()
        # if self.is_first_layer:
        #     return keras.layers.MaxPooling2D(pool_size=_pool_size,
        #                                      padding=_padding,
        #                                      input_shape=self.input_shape,
        #                                      batch_size=self.batch_size)
        # else:
        #     return keras.layers.MaxPooling2D(pool_size=_pool_size,
        #                                      padding=_padding)

    # MaxPooling3D Layer
    def r_max_pooling3d(self, _config=None):
        if self.layer_config:
            return keras.layers.MaxPooling3D(pool_size=_config[0],
                                             padding=_config[1],
                                             input_shape=_config[2],
                                             batch_size=_config[3])
        else:
            _pool_size = random.randint(1, 3)
            _padding = r_padding2()
            return keras.layers.MaxPooling3D(pool_size=_pool_size,
                                             padding=_padding,
                                             input_shape=self.input_shape,
                                             batch_size=self.batch_size), \
                   [_pool_size, _padding, self.input_shape, self.batch_size]
        # if self.layer_config:
        #     return keras.layers.MaxPooling3D.from_config(_config)
        # _pool_size = random.randint(1, 3)
        # _padding = r_padding2()
        # if self.is_first_layer:
        #     return keras.layers.MaxPooling3D(pool_size=_pool_size,
        #                                      padding=_padding,
        #                                      input_shape=self.input_shape,
        #                                      batch_size=self.batch_size)
        # else:
        #     return keras.layers.MaxPooling3D(pool_size=_pool_size,
        #                                      padding=_padding)

    # AveragePooling1D Layer
    def r_average_pooling1d(self, _config=None):
        if self.layer_config:
            return keras.layers.AveragePooling1D(pool_size=_config[0],
                                                 padding=_config[1],
                                                 input_shape=_config[2],
                                                 batch_size=_config[3])
        else:
            _pool_size = random.randint(1, 3)
            _padding = r_padding2()
            return keras.layers.AveragePooling1D(pool_size=_pool_size,
                                                 padding=_padding,
                                                 input_shape=self.input_shape,
                                                 batch_size=self.batch_size), \
                   [_pool_size, _padding, self.input_shape, self.batch_size]
        # if self.layer_config:
        #     return keras.layers.AveragePooling1D.from_config(_config)
        # _pool_size = random.randint(1, 3)
        # _padding = r_padding2()
        # if self.is_first_layer:
        #     return keras.layers.AveragePooling1D(pool_size=_pool_size,
        #                                          padding=_padding,
        #                                          input_shape=self.input_shape,
        #                                          batch_size=self.batch_size)
        # else:
        #     return keras.layers.AveragePooling1D(pool_size=_pool_size,
        #                                          padding=_padding)

    # AveragePooling2D Layer
    def r_average_pooling2d(self, _config=None):
        if self.layer_config:
            return keras.layers.AveragePooling2D(pool_size=_config[0],
                                                 padding=_config[1],
                                                 input_shape=_config[2],
                                                 batch_size=_config[3])
        else:
            _pool_size = random.randint(1, 3)
            _padding = r_padding2()
            return keras.layers.AveragePooling2D(pool_size=_pool_size,
                                                 padding=_padding,
                                                 input_shape=self.input_shape,
                                                 batch_size=self.batch_size), \
                   [_pool_size, _padding, self.input_shape, self.batch_size]
        # if self.layer_config:
        #     return keras.layers.AveragePooling2D.from_config(_config)
        # _pool_size = random.randint(1, 3)
        # _padding = r_padding2()
        # if self.is_first_layer:
        #     return keras.layers.AveragePooling2D(pool_size=_pool_size,
        #                                          padding=_padding,
        #                                          input_shape=self.input_shape,
        #                                          batch_size=self.batch_size)
        # else:
        #     return keras.layers.AveragePooling2D(pool_size=_pool_size,
        #                                          padding=_padding)

    # AveragePooling3D Layer
    def r_average_pooling3d(self, _config=None):
        if self.layer_config:
            return keras.layers.AveragePooling3D(pool_size=_config[0],
                                                 padding=_config[1],
                                                 input_shape=_config[2],
                                                 batch_size=_config[3])
        else:
            _pool_size = random.randint(1, 3)
            _padding = r_padding2()
            return keras.layers.AveragePooling3D(pool_size=_pool_size,
                                                 padding=_padding,
                                                 input_shape=self.input_shape,
                                                 batch_size=self.batch_size), \
                   [_pool_size, _padding, self.input_shape, self.batch_size]
        # if self.layer_config:
        #     return keras.layers.AveragePooling3D.from_config(_config)
        # _pool_size = random.randint(1, 3)
        # _padding = r_padding2()
        # if self.is_first_layer:
        #     return keras.layers.AveragePooling3D(pool_size=_pool_size,
        #                                          padding=_padding,
        #                                          input_shape=self.input_shape,
        #                                          batch_size=self.batch_size)
        # else:
        #     return keras.layers.AveragePooling3D(pool_size=_pool_size,
        #                                          padding=_padding)

    # GlobalMaxPooling1D Layer
    def r_global_max_pooling1d(self, _config=None):
        if self.layer_config:
            return keras.layers.GlobalMaxPooling1D(input_shape=_config[0],
                                                   batch_size=_config[1])
        else:
            return keras.layers.GlobalMaxPooling1D(input_shape=self.input_shape,
                                                   batch_size=self.batch_size), \
                   [self.input_shape, self.batch_size]
        # return keras.layers.GlobalMaxPooling1D(input_shape=self.input_shape,
        #                                        batch_size=self.batch_size)
        # if self.layer_config:
        #     return keras.layers.GlobalMaxPooling1D.from_config(_config)
        # if self.is_first_layer:
        #     return keras.layers.GlobalMaxPooling1D(input_shape=self.input_shape,
        #                                            batch_size=self.batch_size)
        # else:
        #     return keras.layers.GlobalAveragePooling1D()

    # GlobalAveragePooling1D Layer
    def r_global_average_pooling1d(self, _config=None):
        if self.layer_config:
            return keras.layers.GlobalAveragePooling1D(input_shape=self.input_shape,
                                                       batch_size=self.batch_size)
        else:
            return keras.layers.GlobalAveragePooling1D(input_shape=self.input_shape,
                                                       batch_size=self.batch_size), \
                   [self.input_shape, self.batch_size]
        # if self.layer_config:
        #     return keras.layers.GlobalAveragePooling1D.from_config(_config)
        # if self.is_first_layer:
        #     return keras.layers.GlobalAveragePooling1D(input_shape=self.input_shape,
        #                                                batch_size=self.batch_size)
        # else:
        #     return keras.layers.GlobalAveragePooling1D()

    # GlobalMaxPooling2D Layer
    def r_global_max_pooling2d(self, _config=None):
        if self.layer_config:
            return keras.layers.GlobalMaxPooling2D(input_shape=_config[0],
                                                   batch_size=_config[1])
        else:
            return keras.layers.GlobalMaxPooling2D(input_shape=self.input_shape,
                                                   batch_size=self.batch_size), \
                   [self.input_shape, self.batch_size]
        # if self.layer_config:
        #     return keras.layers.GlobalMaxPooling2D.from_config(_config)
        # if self.is_first_layer:
        #     return keras.layers.GlobalMaxPooling2D(input_shape=self.input_shape,
        #                                            batch_size=self.batch_size)
        # else:
        #     return keras.layers.GlobalMaxPooling2D()

    # GlobalAveragePooling2D Layer
    def r_global_average_pooling2d(self, _config=None):
        if self.layer_config:
            return keras.layers.GlobalAveragePooling2D(input_shape=_config[0],
                                                       batch_size=_config[1])
        else:
            return keras.layers.GlobalAveragePooling2D(input_shape=self.input_shape,
                                                       batch_size=self.batch_size), \
                   [self.input_shape, self.batch_size]
        # if self.layer_config:
        #     return keras.layers.GlobalAveragePooling2D.from_config(_config)
        # if self.is_first_layer:
        #     return keras.layers.GlobalAveragePooling2D(input_shape=self.input_shape,
        #                                                batch_size=self.batch_size)
        # else:
        #     return keras.layers.GlobalAveragePooling2D()

    # GlobalMaxPooling3D Layer
    def r_global_max_pooling3d(self, _config=None):
        if self.layer_config:
            return keras.layers.GlobalMaxPooling3D(input_shape=_config[0],
                                                   batch_size=_config[1])
        else:
            return keras.layers.GlobalMaxPooling3D(input_shape=self.input_shape,
                                                   batch_size=self.batch_size), \
                   [self.input_shape, self.batch_size]
        # if self.layer_config:
        #     return keras.layers.GlobalMaxPooling3D.from_config(_config)
        # if self.is_first_layer:
        #     return keras.layers.GlobalMaxPooling3D(input_shape=self.input_shape,
        #                                            batch_size=self.batch_size)
        # else:
        #     return keras.layers.GlobalMaxPooling3D()

    # GlobalAveragePooling3D Layer
    def r_global_average_pooling3d(self, _config=None):
        if self.layer_config:
            return keras.layers.GlobalAveragePooling3D(input_shape=_config[0],
                                                       batch_size=_config[1])
        else:
            return keras.layers.GlobalAveragePooling3D(input_shape=self.input_shape,
                                                       batch_size=self.batch_size), \
                   [self.input_shape, self.batch_size]
        # if self.layer_config:
        #     return keras.layers.GlobalAveragePooling3D.from_config(_config)
        # if self.is_first_layer:
        #     return keras.layers.GlobalAveragePooling3D(input_shape=self.input_shape,
        #                                                batch_size=self.batch_size)
        # else:
        #     return keras.layers.GlobalAveragePooling3D()

    # LocallyConnected1D Layer
    def r_locally_connected1d(self, _config=None):
        if self.layer_config:
            return keras.layers.LocallyConnected1D(_config[0],
                                                   _config[1],
                                                   _config[2],
                                                   padding=_config[3],
                                                   activation=_config[4],
                                                   kernel_initializer=_config[5],
                                                   bias_initializer=_config[6],
                                                   kernel_regularizer=_config[7],
                                                   bias_regularizer=_config[8],
                                                   activity_regularizer=_config[9],
                                                   input_shape=_config[10],
                                                   batch_size=_config[11])
        else:
            _filters = random.randint(1, self.max_filter)
            _kernel_size = random.randint(1, _filters)
            _strides = random.randint(1, _filters - _kernel_size + 1)
            _padding = r_padding2()
            _activation = r_activation()
            _kernel_initializer = r_initializer()
            _bias_initializer = r_initializer()
            _kernel_regularizer = r_regularizers()
            _bias_regularizer = r_regularizers()
            _activity_regularizer = r_regularizers()
            return keras.layers.LocallyConnected1D(_filters,
                                                   _kernel_size,
                                                   _strides,
                                                   padding=_padding,
                                                   activation=_activation,
                                                   kernel_initializer=_kernel_initializer,
                                                   bias_initializer=_bias_initializer,
                                                   kernel_regularizer=_kernel_regularizer,
                                                   bias_regularizer=_bias_regularizer,
                                                   activity_regularizer=_activity_regularizer,
                                                   input_shape=self.input_shape,
                                                   batch_size=self.batch_size), \
                   [_filters, _kernel_size, _strides, _padding, _activation, _kernel_initializer, _bias_initializer,
                    _kernel_regularizer, _bias_regularizer, _activity_regularizer, self.input_shape, self.batch_size]
        # if self.layer_config:
        #     return keras.layers.LocallyConnected1D.from_config(_config)
        # _filters = random.randint(1, self.max_filter)
        # _kernel_size = random.randint(1, _filters)
        # _strides = random.randint(1, _filters - _kernel_size + 1)
        # _padding = r_padding2()
        # _activation = r_activation()
        # _kernel_initializer = r_initializer()
        # _bias_initializer = r_initializer()
        # _kernel_regularizer = r_regularizers()
        # _bias_regularizer = r_regularizers()
        # _activity_regularizer = r_regularizers()
        # if self.is_first_layer:
        #     return keras.layers.LocallyConnected1D(_filters,
        #                                            _kernel_size,
        #                                            _strides,
        #                                            padding=_padding,
        #                                            activation=_activation,
        #                                            kernel_initializer=_kernel_initializer,
        #                                            bias_initializer=_bias_initializer,
        #                                            kernel_regularizer=_kernel_regularizer,
        #                                            bias_regularizer=_bias_regularizer,
        #                                            activity_regularizer=_activity_regularizer,
        #                                            input_shape=self.input_shape,
        #                                            batch_size=self.batch_size)
        # else:
        #     return keras.layers.LocallyConnected1D(_filters,
        #                                            _kernel_size,
        #                                            _strides,
        #                                            padding=_padding,
        #                                            activation=_activation,
        #                                            kernel_initializer=_kernel_initializer,
        #                                            bias_initializer=_bias_initializer,
        #                                            kernel_regularizer=_kernel_regularizer,
        #                                            bias_regularizer=_bias_regularizer,
        #                                            activity_regularizer=_activity_regularizer)

    # LocallyConnected2D Layer
    def r_locally_connected2d(self, _config=None):
        if self.layer_config:
            return keras.layers.LocallyConnected2D(_config[0],
                                                   _config[1],
                                                   strides=_config[2],
                                                   padding=_config[3],
                                                   activation=_config[4],
                                                   kernel_initializer=_config[5],
                                                   bias_initializer=_config[6],
                                                   kernel_regularizer=_config[7],
                                                   bias_regularizer=_config[8],
                                                   activity_regularizer=_config[9],
                                                   input_shape=_config[10],
                                                   batch_size=_config[11])
        else:
            _filters = random.randint(1, self.max_filter)
            _kernel_size = random.randint(1, _filters)
            _strides = random.randint(1, _filters - _kernel_size + 1)
            _activation = r_activation()
            _kernel_initializer = r_initializer()
            _bias_initializer = r_initializer()
            _kernel_regularizer = r_regularizers()
            _bias_regularizer = r_regularizers()
            _activity_regularizer = r_regularizers()
            return keras.layers.LocallyConnected2D(_filters,
                                                   (_kernel_size, _kernel_size),
                                                   strides=(_strides, _strides),
                                                   padding='valid',
                                                   activation=_activation,
                                                   kernel_initializer=_kernel_initializer,
                                                   bias_initializer=_bias_initializer,
                                                   kernel_regularizer=_kernel_regularizer,
                                                   bias_regularizer=_bias_regularizer,
                                                   activity_regularizer=_activity_regularizer,
                                                   input_shape=self.input_shape,
                                                   batch_size=self.batch_size), \
                   [_filters, (_kernel_size, _kernel_size), (_strides, _strides), 'valid', _activation, _kernel_initializer,
                    _bias_initializer, _kernel_regularizer, _bias_regularizer, _activity_regularizer, self.input_shape,
                    self.batch_size]
        # if self.layer_config:
        #     return keras.layers.LocallyConnected2D.from_config(_config)
        # _filters = random.randint(1, self.max_filter)
        # _kernel_size = random.randint(1, _filters)
        # _strides = random.randint(1, _filters - _kernel_size + 1)
        # _activation = r_activation()
        # _kernel_initializer = r_initializer()
        # _bias_initializer = r_initializer()
        # _kernel_regularizer = r_regularizers()
        # _bias_regularizer = r_regularizers()
        # _activity_regularizer = r_regularizers()
        # if self.is_first_layer:
        #     return keras.layers.LocallyConnected2D(_filters,
        #                                            (_kernel_size, _kernel_size),
        #                                            strides=(_strides, _strides),
        #                                            padding='valid',
        #                                            activation=_activation,
        #                                            kernel_initializer=_kernel_initializer,
        #                                            bias_initializer=_bias_initializer,
        #                                            kernel_regularizer=_kernel_regularizer,
        #                                            bias_regularizer=_bias_regularizer,
        #                                            activity_regularizer=_activity_regularizer,
        #                                            input_shape=self.input_shape,
        #                                            batch_size=self.batch_size)
        # else:
        #     return keras.layers.LocallyConnected2D(_filters,
        #                                            (_kernel_size, _kernel_size),
        #                                            strides=(_strides, _strides),
        #                                            padding='valid',
        #                                            activation=_activation,
        #                                            kernel_initializer=_kernel_initializer,
        #                                            bias_initializer=_bias_initializer,
        #                                            kernel_regularizer=_kernel_regularizer,
        #                                            bias_regularizer=_bias_regularizer,
        #                                            activity_regularizer=_activity_regularizer)

    # SimpleRNN Layer
    def r_simple_rnn(self, _config=None):
        if self.layer_config:
            return keras.layers.SimpleRNN(_config[0],
                                          activation=_config[1],
                                          kernel_initializer=_config[2],
                                          recurrent_initializer=_config[3],
                                          bias_initializer=_config[4],
                                          kernel_regularizer=_config[5],
                                          recurrent_regularizer=_config[6],
                                          bias_regularizer=_config[7],
                                          activity_regularizer=_config[8],
                                          dropout=_config[9],
                                          recurrent_dropout=_config[10],
                                          return_sequences=_config[11],
                                          return_state=_config[12],
                                          go_backwards=_config[13],
                                          stateful=_config[13],
                                          unroll=_config[14],
                                          input_shape=_config[15],
                                          batch_size=_config[16])
        else:
            _units = random.randint(1, self.max_units)
            _activation = r_activation()
            _kernel_initializer = r_initializer()
            _recurrent_initializer = r_initializer()
            _bias_initializer = r_initializer()
            _kernel_regularizer = r_regularizers()
            _recurrent_regularizer = r_regularizers()
            _bias_regularizer = r_regularizers()
            _activity_regularizer = r_regularizers()
            _dropout = random.uniform(0., self.max_dropout_rate)
            _recurent_dropout = random.uniform(0., self.max_dropout_rate)
            _return_sequences = False
            _return_state = False
            _go_backwards = False
            _stateful = False
            _unroll = False
            return keras.layers.SimpleRNN(_units,
                                          activation=_activation,
                                          kernel_initializer=_kernel_initializer,
                                          recurrent_initializer=_recurrent_initializer,
                                          bias_initializer=_bias_initializer,
                                          kernel_regularizer=_kernel_regularizer,
                                          recurrent_regularizer=_recurrent_regularizer,
                                          bias_regularizer=_bias_regularizer,
                                          activity_regularizer=_activity_regularizer,
                                          dropout=_dropout,
                                          recurrent_dropout=_recurent_dropout,
                                          return_sequences=_return_sequences,
                                          return_state=_return_state,
                                          go_backwards=_go_backwards,
                                          stateful=_stateful,
                                          unroll=_unroll,
                                          input_shape=self.input_shape,
                                          batch_size=self.batch_size), \
                   [_units, _activation, _kernel_initializer, _recurrent_initializer, _bias_initializer,
                    _kernel_regularizer, _recurrent_regularizer, _bias_regularizer, _activity_regularizer, _dropout,
                    _recurent_dropout, _return_sequences, _return_state, _go_backwards, _stateful, _unroll,
                    self.input_shape, self.batch_size]
        # if self.layer_config:
        #     return keras.layers.SimpleRNN.from_config(_config)
        # _units = random.randint(1, self.max_units)
        # _activation = r_activation()
        # _kernel_initializer = r_initializer()
        # _recurrent_initializer = r_initializer()
        # _bias_initializer = r_initializer()
        # _kernel_regularizer = r_regularizers()
        # _recurrent_regularizer = r_regularizers()
        # _bias_regularizer = r_regularizers()
        # _activity_regularizer = r_regularizers()
        # _dropout = random.uniform(0., self.max_dropout_rate)
        # _recurent_dropout = random.uniform(0., self.max_dropout_rate)
        # _return_sequences = False
        # _return_state = False
        # _go_backwards = False
        # _stateful = False
        # _unroll = False
        # if self.is_first_layer:
        #     return keras.layers.SimpleRNN(_units,
        #                                   activation=_activation,
        #                                   kernel_initializer=_kernel_initializer,
        #                                   recurrent_initializer=_recurrent_initializer,
        #                                   bias_initializer=_bias_initializer,
        #                                   kernel_regularizer=_kernel_regularizer,
        #                                   recurrent_regularizer=_recurrent_regularizer,
        #                                   bias_regularizer=_bias_regularizer,
        #                                   activity_regularizer=_activity_regularizer,
        #                                   dropout=_dropout,
        #                                   recurrent_dropout=_recurent_dropout,
        #                                   return_sequences=_return_sequences,
        #                                   return_state=_return_state,
        #                                   go_backwards=_go_backwards,
        #                                   stateful=_stateful,
        #                                   unroll=_unroll,
        #                                   input_shape=self.input_shape,
        #                                   batch_size=self.batch_size)
        # else:
        #     return keras.layers.SimpleRNN(_units,
        #                                   activation=_activation,
        #                                   kernel_initializer=_kernel_initializer,
        #                                   recurrent_initializer=_recurrent_initializer,
        #                                   bias_initializer=_bias_initializer,
        #                                   kernel_regularizer=_kernel_regularizer,
        #                                   recurrent_regularizer=_recurrent_regularizer,
        #                                   bias_regularizer=_bias_regularizer,
        #                                   activity_regularizer=_activity_regularizer,
        #                                   dropout=_dropout,
        #                                   recurrent_dropout=_recurent_dropout,
        #                                   return_sequences=_return_sequences,
        #                                   return_state=_return_state,
        #                                   go_backwards=_go_backwards,
        #                                   stateful=_stateful,
        #                                   unroll=_unroll)

    # GRU Layer
    def r_gru(self, _config=None):
        if self.layer_config:
            return keras.layers.GRU(_config[0],
                                    activation=_config[1],
                                    recurrent_activation=_config[2],
                                    kernel_initializer=_config[3],
                                    recurrent_initializer=_config[4],
                                    bias_initializer=_config[5],
                                    kernel_regularizer=_config[6],
                                    recurrent_regularizer=_config[7],
                                    bias_regularizer=_config[8],
                                    activity_regularizer=_config[9],
                                    dropout=_config[10],
                                    recurrent_dropout=_config[11],
                                    implementation=_config[12],
                                    return_sequences=_config[13],
                                    return_state=_config[14],
                                    go_backwards=_config[15],
                                    stateful=_config[16],
                                    unroll=_config[17],
                                    input_shape=_config[18],
                                    batch_size=_config[19]
                                    )
        else:
            _units = random.randint(1, self.max_units)
            _activation = r_activation()
            _recurrent_activation = r_activation()
            _kernel_initializer = r_initializer()
            _recurrent_initializer = r_initializer()
            _bias_initializer = r_initializer()
            _kernel_regularizer = r_regularizers()
            _recurrent_regularizer = r_regularizers()
            _bias_regularizer = r_regularizers()
            _activity_regularizer = r_regularizers()
            _dropout = random.uniform(0., self.max_dropout_rate)
            _recurent_dropout = random.uniform(0., self.max_dropout_rate)
            _implementation = random.randint(1, 2)
            _return_sequences = False
            _return_state = False
            _go_backwards = False
            _stateful = False
            _unroll = False
            return keras.layers.GRU(_units,
                                    activation=_activation,
                                    recurrent_activation=_recurrent_activation,
                                    kernel_initializer=_kernel_initializer,
                                    recurrent_initializer=_recurrent_initializer,
                                    bias_initializer=_bias_initializer,
                                    kernel_regularizer=_kernel_regularizer,
                                    recurrent_regularizer=_recurrent_regularizer,
                                    bias_regularizer=_bias_regularizer,
                                    activity_regularizer=_activity_regularizer,
                                    dropout=_dropout,
                                    recurrent_dropout=_recurent_dropout,
                                    implementation=_implementation,
                                    return_sequences=_return_sequences,
                                    return_state=_return_state,
                                    go_backwards=_go_backwards,
                                    stateful=_stateful,
                                    unroll=_unroll,
                                    input_shape=self.input_shape,
                                    batch_size=self.batch_size
                                    ), \
                   [_units, _activation, _recurrent_activation, _kernel_initializer, _recurrent_initializer,
                    _bias_initializer, _kernel_regularizer, _recurrent_regularizer, _bias_regularizer, _activity_regularizer,
                    _dropout, _recurent_dropout, _implementation, _return_sequences, _return_state, _go_backwards,
                    _stateful, _unroll, self.input_shape, self.batch_size]
        # if self.layer_config:
        #     return keras.layers.GRU.from_config(_config)
        # _units = random.randint(1, self.max_units)
        # _activation = r_activation()
        # _recurrent_activation = r_activation()
        # _kernel_initializer = r_initializer()
        # _recurrent_initializer = r_initializer()
        # _bias_initializer = r_initializer()
        # _kernel_regularizer = r_regularizers()
        # _recurrent_regularizer = r_regularizers()
        # _bias_regularizer = r_regularizers()
        # _activity_regularizer = r_regularizers()
        # _dropout = random.uniform(0., self.max_dropout_rate)
        # _recurent_dropout = random.uniform(0., self.max_dropout_rate)
        # _implementation = random.randint(1, 2)
        # _return_sequences = False
        # _return_state = False
        # _go_backwards = False
        # _stateful = False
        # _unroll = False
        # if self.is_first_layer:
        #     return keras.layers.GRU(_units,
        #                             activation=_activation,
        #                             recurrent_activation=_recurrent_activation,
        #                             kernel_initializer=_kernel_initializer,
        #                             recurrent_initializer=_recurrent_initializer,
        #                             bias_initializer=_bias_initializer,
        #                             kernel_regularizer=_kernel_regularizer,
        #                             recurrent_regularizer=_recurrent_regularizer,
        #                             bias_regularizer=_bias_regularizer,
        #                             activity_regularizer=_activity_regularizer,
        #                             dropout=_dropout,
        #                             recurrent_dropout=_recurent_dropout,
        #                             implementation=_implementation,
        #                             return_sequences=_return_sequences,
        #                             return_state=_return_state,
        #                             go_backwards=_go_backwards,
        #                             stateful=_stateful,
        #                             unroll=_unroll,
        #                             input_shape=self.input_shape,
        #                             batch_size=self.batch_size
        #                             )
        # else:
        #     return keras.layers.GRU(_units,
        #                             activation=_activation,
        #                             recurrent_activation=_recurrent_activation,
        #                             kernel_initializer=_kernel_initializer,
        #                             recurrent_initializer=_recurrent_initializer,
        #                             bias_initializer=_bias_initializer,
        #                             kernel_regularizer=_kernel_regularizer,
        #                             recurrent_regularizer=_recurrent_regularizer,
        #                             bias_regularizer=_bias_regularizer,
        #                             activity_regularizer=_activity_regularizer,
        #                             dropout=_dropout,
        #                             recurrent_dropout=_recurent_dropout,
        #                             implementation=_implementation,
        #                             return_sequences=_return_sequences,
        #                             return_state=_return_state,
        #                             go_backwards=_go_backwards,
        #                             stateful=_stateful,
        #                             unroll=_unroll
        #                             )

    # LSTM Layer
    def r_lstm(self, _config=None):
        if self.layer_config:
            return keras.layers.LSTM(_config[0],
                                     activation=_config[1],
                                     recurrent_activation=_config[2],
                                     kernel_initializer=_config[3],
                                     recurrent_initializer=_config[4],
                                     bias_initializer=_config[5],
                                     unit_forget_bias=_config[6],
                                     kernel_regularizer=_config[7],
                                     recurrent_regularizer=_config[8],
                                     bias_regularizer=_config[9],
                                     activity_regularizer=_config[10],
                                     dropout=_config[11],
                                     recurrent_dropout=_config[12],
                                     implementation=_config[13],
                                     return_sequences=_config[14],
                                     return_state=_config[15],
                                     go_backwards=_config[16],
                                     stateful=_config[17],
                                     unroll=_config[18],
                                     input_shape=_config[19],
                                     batch_size=_config[20]
                                     )
        else:
            _units = random.randint(1, self.max_units)
            _activation = r_activation()
            _recurrent_activation = r_activation()
            _kernel_initializer = r_initializer()
            _recurrent_initializer = r_initializer()
            _bias_initializer = r_initializer()
            _unit_forget_bias = random.sample([True, False], 1)[0]
            _kernel_regularizer = r_regularizers()
            _recurrent_regularizer = r_regularizers()
            _bias_regularizer = r_regularizers()
            _activity_regularizer = r_regularizers()
            _dropout = random.uniform(0., self.max_dropout_rate)
            _recurent_dropout = random.uniform(0., self.max_dropout_rate)
            _implementation = random.randint(1, 2)
            _return_sequences = False
            _return_state = False
            _go_backwards = False
            _stateful = False
            _unroll = False
            return keras.layers.LSTM(_units,
                                     activation=_activation,
                                     recurrent_activation=_recurrent_activation,
                                     kernel_initializer=_kernel_initializer,
                                     recurrent_initializer=_recurrent_initializer,
                                     bias_initializer=_bias_initializer,
                                     unit_forget_bias=_unit_forget_bias,
                                     kernel_regularizer=_kernel_regularizer,
                                     recurrent_regularizer=_recurrent_regularizer,
                                     bias_regularizer=_bias_regularizer,
                                     activity_regularizer=_activity_regularizer,
                                     dropout=_dropout,
                                     recurrent_dropout=_recurent_dropout,
                                     implementation=_implementation,
                                     return_sequences=_return_sequences,
                                     return_state=_return_state,
                                     go_backwards=_go_backwards,
                                     stateful=_stateful,
                                     unroll=_unroll,
                                     input_shape=self.input_shape,
                                     batch_size=self.batch_size
                                     ), \
                   [_units, _activation, _recurrent_activation, _kernel_initializer, _recurrent_initializer,
                    _bias_initializer, _unit_forget_bias, _kernel_regularizer, _recurrent_regularizer, _bias_regularizer,
                    _activity_regularizer, _dropout, _recurent_dropout, _implementation, _return_sequences, _return_state,
                    _go_backwards, _stateful, _unroll, self.input_shape, self.batch_size]
        # if self.layer_config:
        #     return keras.layers.LSTM.from_config(_config)
        # _units = random.randint(1, self.max_units)
        # _activation = r_activation()
        # _recurrent_activation = r_activation()
        # _kernel_initializer = r_initializer()
        # _recurrent_initializer = r_initializer()
        # _bias_initializer = r_initializer()
        # _unit_forget_bias = random.sample([True, False], 1)[0]
        # _kernel_regularizer = r_regularizers()
        # _recurrent_regularizer = r_regularizers()
        # _bias_regularizer = r_regularizers()
        # _activity_regularizer = r_regularizers()
        # _dropout = random.uniform(0., self.max_dropout_rate)
        # _recurent_dropout = random.uniform(0., self.max_dropout_rate)
        # _implementation = random.randint(1, 2)
        # _return_sequences = False
        # _return_state = False
        # _go_backwards = False
        # _stateful = False
        # _unroll = False
        # if self.is_first_layer:
        #     return keras.layers.LSTM(_units,
        #                              activation=_activation,
        #                              recurrent_activation=_recurrent_activation,
        #                              kernel_initializer=_kernel_initializer,
        #                              recurrent_initializer=_recurrent_initializer,
        #                              bias_initializer=_bias_initializer,
        #                              unit_forget_bias=_unit_forget_bias,
        #                              kernel_regularizer=_kernel_regularizer,
        #                              recurrent_regularizer=_recurrent_regularizer,
        #                              bias_regularizer=_bias_regularizer,
        #                              activity_regularizer=_activity_regularizer,
        #                              dropout=_dropout,
        #                              recurrent_dropout=_recurent_dropout,
        #                              implementation=_implementation,
        #                              return_sequences=_return_sequences,
        #                              return_state=_return_state,
        #                              go_backwards=_go_backwards,
        #                              stateful=_stateful,
        #                              unroll=_unroll,
        #                              input_shape=self.input_shape,
        #                              batch_size=self.batch_size
        #                              )
        # else:
        #     return keras.layers.LSTM(_units,
        #                              activation=_activation,
        #                              recurrent_activation=_recurrent_activation,
        #                              kernel_initializer=_kernel_initializer,
        #                              recurrent_initializer=_recurrent_initializer,
        #                              bias_initializer=_bias_initializer,
        #                              unit_forget_bias=_unit_forget_bias,
        #                              kernel_regularizer=_kernel_regularizer,
        #                              recurrent_regularizer=_recurrent_regularizer,
        #                              bias_regularizer=_bias_regularizer,
        #                              activity_regularizer=_activity_regularizer,
        #                              dropout=_dropout,
        #                              recurrent_dropout=_recurent_dropout,
        #                              implementation=_implementation,
        #                              return_sequences=_return_sequences,
        #                              return_state=_return_state,
        #                              go_backwards=_go_backwards,
        #                              stateful=_stateful,
        #                              unroll=_unroll
        #                              )

    # ConvLSTM2D Layer
    def r_conv_lstm2d(self, _config=None):
        if self.layer_config:
            return keras.layers.ConvLSTM2D(_config[0],
                                           _config[1],
                                           strides=_config[2],
                                           padding=_config[3],
                                           activation=_config[4],
                                           recurrent_activation=_config[5],
                                           kernel_initializer=_config[6],
                                           recurrent_initializer=_config[7],
                                           bias_initializer=_config[8],
                                           unit_forget_bias=_config[9],
                                           kernel_regularizer=_config[10],
                                           recurrent_regularizer=_config[11],
                                           bias_regularizer=_config[12],
                                           activity_regularizer=_config[13],
                                           return_sequences=_config[14],
                                           go_backwards=_config[15],
                                           stateful=_config[16],
                                           dropout=_config[17],
                                           recurrent_dropout=_config[18],
                                           input_shape=_config[19],
                                           batch_size=_config[20])
        else:
            _filters = random.randint(1, self.max_filter)
            _kernel_size = random.randint(1, _filters)
            _strides = random.randint(1, _filters - _kernel_size + 1)
            _padding = r_padding2()
            _activation = r_activation()
            _recurrent_activation = r_activation()
            _kernel_initializer = r_initializer()
            _recurrent_initializer = r_initializer()
            _bias_initializer = r_initializer()
            _unit_forget_bias = random.choice([True, False])
            _kernel_regularizer = r_regularizers()
            _recurrent_regularizer = r_regularizers()
            _bias_regularizer = r_regularizers()
            _activity_regularizer = r_regularizers()
            _return_sequences = random.choice([True, False])
            _go_backwards = random.choice([True, False])
            _stateful = False
            _dropout = random.uniform(0., self.max_dropout_rate)
            _recurent_dropout = random.uniform(0., self.max_dropout_rate)
            return keras.layers.ConvLSTM2D(_filters,
                                           (_kernel_size, _kernel_size),
                                           strides=(_strides, _strides),
                                           padding=_padding,
                                           activation=_activation,
                                           recurrent_activation=_recurrent_activation,
                                           kernel_initializer=_kernel_initializer,
                                           recurrent_initializer=_recurrent_initializer,
                                           bias_initializer=_bias_initializer,
                                           unit_forget_bias=_unit_forget_bias,
                                           kernel_regularizer=_kernel_regularizer,
                                           recurrent_regularizer=_recurrent_regularizer,
                                           bias_regularizer=_bias_regularizer,
                                           activity_regularizer=_activity_regularizer,
                                           return_sequences=_return_sequences,
                                           go_backwards=_go_backwards,
                                           stateful=_stateful,
                                           dropout=_dropout,
                                           recurrent_dropout=_recurent_dropout,
                                           input_shape=self.input_shape,
                                           batch_size=self.batch_size), \
                   [_filters, (_kernel_size, _kernel_size), (_strides, _strides), _padding, _activation, _recurrent_activation,
                    _kernel_initializer, _recurrent_initializer, _bias_initializer, _unit_forget_bias, _kernel_regularizer,
                    _recurrent_regularizer, _bias_regularizer, _activity_regularizer, _return_sequences, _go_backwards,
                    _stateful, _dropout, _recurent_dropout, self.input_shape, self.batch_size]
        # if self.layer_config:
        #     return keras.layers.ConvLSTM2D.from_config(_config)
        # _filters = random.randint(1, self.max_filter)
        # _kernel_size = random.randint(1, _filters)
        # _strides = random.randint(1, _filters - _kernel_size + 1)
        # _padding = r_padding2()
        # _activation = r_activation()
        # _recurrent_activation = r_activation()
        # _kernel_initializer = r_initializer()
        # _recurrent_initializer = r_initializer()
        # _bias_initializer = r_initializer()
        # _unit_forget_bias = random.choice([True, False])
        # _kernel_regularizer = r_regularizers()
        # _recurrent_regularizer = r_regularizers()
        # _bias_regularizer = r_regularizers()
        # _activity_regularizer = r_regularizers()
        # _return_sequences = random.choice([True, False])
        # _go_backwards = random.choice([True, False])
        # _stateful = False
        # _dropout = random.uniform(0., self.max_dropout_rate)
        # _recurent_dropout = random.uniform(0., self.max_dropout_rate)
        # if self.is_first_layer:
        #     return keras.layers.ConvLSTM2D(_filters,
        #                                    (_kernel_size, _kernel_size),
        #                                    strides=(_strides, _strides),
        #                                    padding=_padding,
        #                                    activation=_activation,
        #                                    recurrent_activation=_recurrent_activation,
        #                                    kernel_initializer=_kernel_initializer,
        #                                    recurrent_initializer=_recurrent_initializer,
        #                                    bias_initializer=_bias_initializer,
        #                                    unit_forget_bias=_unit_forget_bias,
        #                                    kernel_regularizer=_kernel_regularizer,
        #                                    recurrent_regularizer=_recurrent_regularizer,
        #                                    bias_regularizer=_bias_regularizer,
        #                                    activity_regularizer=_activity_regularizer,
        #                                    return_sequences=_return_sequences,
        #                                    go_backwards=_go_backwards,
        #                                    stateful=_stateful,
        #                                    dropout=_dropout,
        #                                    recurrent_dropout=_recurent_dropout,
        #                                    input_shape=self.input_shape,
        #                                    batch_size=self.batch_size)
        # else:
        #     return keras.layers.ConvLSTM2D(_filters,
        #                                    (_kernel_size, _kernel_size),
        #                                    strides=(_strides, _strides),
        #                                    padding=_padding,
        #                                    activation=_activation,
        #                                    recurrent_activation=_recurrent_activation,
        #                                    kernel_initializer=_kernel_initializer,
        #                                    recurrent_initializer=_recurrent_initializer,
        #                                    bias_initializer=_bias_initializer,
        #                                    unit_forget_bias=_unit_forget_bias,
        #                                    kernel_regularizer=_kernel_regularizer,
        #                                    recurrent_regularizer=_recurrent_regularizer,
        #                                    bias_regularizer=_bias_regularizer,
        #                                    activity_regularizer=_activity_regularizer,
        #                                    return_sequences=_return_sequences,
        #                                    go_backwards=_go_backwards,
        #                                    stateful=_stateful,
        #                                    dropout=_dropout,
        #                                    recurrent_dropout=_recurent_dropout
        #                                    )

    # Embedding Layer
    def r_embedding(self, _config=None):
        if self.layer_config:
            return keras.layers.Embedding(_config[0],
                                          _config[1],
                                          embeddings_initializer=_config[2],
                                          embeddings_regularizer=_config[3],
                                          activity_regularizer=_config[4],
                                          mask_zero=_config[5],
                                          input_length=_config[6])
        else:
            _input_dim = random.randint(1, self.max_input_dim)
            _output_dim = random.randint(1, self.max_output_dim)
            _embeddings_initializer = r_initializer()
            _embeddings_regularizer = r_regularizers()
            _activity_regularizer = r_regularizers()
            # _mask_zero = random.sample([True, False], 1)[0]
            _mask_zero = random.choice([True, False])
            _input_length = self.input_shape[0]
            return keras.layers.Embedding(_input_dim,
                                          _output_dim,
                                          embeddings_initializer=_embeddings_initializer,
                                          embeddings_regularizer=_embeddings_regularizer,
                                          activity_regularizer=_activity_regularizer,
                                          mask_zero=_mask_zero,
                                          input_length=_input_length), \
                   [_input_dim, _output_dim, _embeddings_initializer, _embeddings_regularizer, _activity_regularizer,
                    _mask_zero, _input_length]

        # if self.layer_config:
        #     return keras.layers.Embedding.from_config(_config)
        # _input_dim = random.randint(1, self.max_input_dim)
        # _output_dim = random.randint(1, self.max_output_dim)
        # _embeddings_initializer = r_initializer()
        # _embeddings_regularizer = r_regularizers()
        # _activity_regularizer = r_regularizers()
        # # _mask_zero = random.sample([True, False], 1)[0]
        # _mask_zero = random.choice([True, False])
        # _input_length = self.input_shape[0]
        # return keras.layers.Embedding(_input_dim,
        #                               _output_dim,
        #                               embeddings_initializer=_embeddings_initializer,
        #                               embeddings_regularizer=_embeddings_regularizer,
        #                               activity_regularizer=_activity_regularizer,
        #                               mask_zero=_mask_zero,
        #                               input_length=_input_length)

    # LeakyReLU Layer
    def r_leaky_relu(self, _config=None):
        if self.layer_config:
            return keras.layers.LeakyReLU(alpha=_config[0],
                                          input_shape=_config[1],
                                          batch_size=_config[2])
        else:
            _alpha = random.uniform(0., 1.0)
            return keras.layers.LeakyReLU(alpha=_alpha,
                                          input_shape=self.input_shape,
                                          batch_size=self.batch_size), \
                   [_alpha, self.input_shape, self.batch_size]
        # if self.layer_config:
        #     return keras.layers.LeakyReLU.from_config(_config)
        # _alpha = random.uniform(0., 1.0)
        # if self.is_first_layer:
        #     return keras.layers.LeakyReLU(alpha=_alpha,
        #                                   input_shape=self.input_shape,
        #                                   batch_size=self.batch_size)
        # else:
        #     return keras.layers.LeakyReLU(alpha=_alpha)

    # PReLU Layer
    def r_p_relu(self, _config=None):
        if self.layer_config:
            return keras.layers.PReLU(alpha_initializer=_config[0],
                                      alpha_regularizer=_config[1],
                                      input_shape=_config[2],
                                      batch_size=_config[3])
        else:
            _alpha_initializer = r_initializer()
            _alpha_regularizer = r_regularizers()
            return keras.layers.PReLU(alpha_initializer=_alpha_initializer,
                                      alpha_regularizer=_alpha_regularizer,
                                      input_shape=self.input_shape,
                                      batch_size=self.batch_size), \
                   [_alpha_initializer, _alpha_regularizer, self.input_shape, self.batch_size]
        # if self.layer_config:
        #     return keras.layers.PReLU.from_config(_config)
        # _alpha_initializer = r_initializer()
        # _alpha_regularizer = r_regularizers()
        # if self.is_first_layer:
        #     return keras.layers.PReLU(alpha_initializer=_alpha_initializer,
        #                               alpha_regularizer=_alpha_regularizer,
        #                               input_shape=self.input_shape,
        #                               batch_size=self.batch_size)
        # else:
        #     return keras.layers.PReLU(alpha_initializer=_alpha_initializer,
        #                               alpha_regularizer=_alpha_regularizer)

    # ELU Layer
    def r_elu(self, _config=None):
        if self.layer_config:
            return keras.layers.ELU(alpha=_config[0],
                                    input_shape=_config[1],
                                    batch_size=_config[2])
        else:
            _alpha = random.uniform(0., 1.0)
            return keras.layers.ELU(alpha=_alpha,
                                    input_shape=self.input_shape,
                                    batch_size=self.batch_size), \
                   [_alpha, self.input_shape, self.batch_size]
        # if self.layer_config:
        #     return keras.layers.ELU.from_config(_config)
        # _alpha = random.uniform(0., 1.0)
        # if self.is_first_layer:
        #     return keras.layers.ELU(alpha=_alpha,
        #                             input_shape=self.input_shape,
        #                             batch_size=self.batch_size)
        # else:
        #     return keras.layers.ELU(alpha=_alpha)

    # ThresholdedReLU Layer
    def r_thresholded_relu(self, _config=None):
        if self.layer_config:
            return keras.layers.ThresholdedReLU(theta=_config[0],
                                                input_shape=_config[1],
                                                batch_size=_config[2])
        else:
            _theta = random.uniform(0., 1.0)
            return keras.layers.ThresholdedReLU(theta=_theta,
                                                input_shape=self.input_shape,
                                                batch_size=self.batch_size), \
                   [_theta, self.input_shape, self.batch_size]
        # if self.layer_config:
        #     return keras.layers.ThresholdedReLU.from_config(_config)
        # _theta = random.uniform(0., 1.0)
        # if self.is_first_layer:
        #     return keras.layers.ThresholdedReLU(theta=_theta,
        #                                         input_shape=self.input_shape,
        #                                         batch_size=self.batch_size)
        # else:
        #     return keras.layers.ThresholdedReLU(theta=_theta)

    # Softmax Layer
    def r_softmax(self, _config=None):
        if self.layer_config:
            return keras.layers.Softmax(input_shape=_config[0],
                                        batch_size=_config[1])
        else:
            return keras.layers.Softmax(input_shape=self.input_shape,
                                        batch_size=self.batch_size), \
                   [self.input_shape, self.batch_size]
        # if self.is_first_layer:
        #     return keras.layers.Softmax(input_shape=self.input_shape,
        #                                 batch_size=self.batch_size)
        # else:
        #     return keras.layers.Softmax()

    # ReLU Layer
    def r_relu(self, _config=None):
        if self.layer_config:
            return keras.layers.ReLU(input_shape=_config[0],
                                     batch_size=_config[1])
        else:
            return keras.layers.ReLU(input_shape=self.input_shape,
                                     batch_size=self.batch_size), \
                   [self.input_shape, self.batch_size]

        # if self.is_first_layer:
        #     return keras.layers.ReLU(input_shape=self.input_shape,
        #                              batch_size=self.batch_size)
        # else:
        #     return keras.layers.ReLU()

    # GaussianNoise Layer
    def r_gaussian_noise(self, _config=None):
        if self.layer_config:
            return keras.layers.GaussianNoise(stddev=_config[0],
                                              input_shape=_config[1],
                                              batch_size=_config[2])
        else:
            _stddev = random.uniform(0., 1.)
            return keras.layers.GaussianNoise(stddev=_stddev,
                                              input_shape=self.input_shape,
                                              batch_size=self.batch_size), \
                   [_stddev, self.input_shape, self.batch_size]
        # if self.layer_config:
        #     return keras.layers.GaussianNoise.from_config(_config)
        # _stddev = random.uniform(0., 1.)
        # if self.is_first_layer:
        #     return keras.layers.GaussianNoise(stddev=_stddev,
        #                                       input_shape=self.input_shape,
        #                                       batch_size=self.batch_size)
        # else:
        #     return keras.layers.GaussianNoise(stddev=_stddev)

    # GaussianDropout Layer
    def r_gaussian_dropout(self, _config=None):
        if self.layer_config:
            return keras.layers.GaussianDropout(rate=_config[0],
                                                input_shape=_config[1],
                                                batch_size=_config[2])
        else:
            _rate = random.uniform(0., self.max_dropout_rate)
            return keras.layers.GaussianDropout(rate=_rate,
                                                input_shape=self.input_shape,
                                                batch_size=self.batch_size), \
                   [_rate, self.input_shape, self.batch_size]
        # if self.layer_config:
        #     return keras.layers.GaussianDropout.from_config(_config)
        # _rate = random.uniform(0., self.max_dropout_rate)
        # if self.is_first_layer:
        #     return keras.layers.GaussianDropout(rate=_rate,
        #                                         input_shape=self.input_shape,
        #                                         batch_size=self.batch_size)
        # else:
        #     return keras.layers.GaussianDropout(rate=_rate)

    # AlphaDropout Layer
    def r_alpha_dropout(self, _config=None):
        if self.layer_config:
            return keras.layers.AlphaDropout(_config[0],
                                             seed=_config[1],
                                             input_shape=_config[2],
                                             batch_size=_config[3])
        else:
            _rate = random.uniform(0., self.max_dropout_rate)
            _seed = random.randint(1, 10)
            return keras.layers.AlphaDropout(_rate,
                                             seed=_seed,
                                             input_shape=self.input_shape,
                                             batch_size=self.batch_size), \
                   [_rate, _seed, self.input_shape, self.batch_size]
        # if self.layer_config:
        #     return keras.layers.AlphaDropout.from_config(_config)
        # _rate = random.uniform(0., self.max_dropout_rate)
        # _seed = random.randint(1, 10)
        # if self.is_first_layer:
        #     return keras.layers.AlphaDropout(_rate,
        #                                      seed=_seed,
        #                                      input_shape=self.input_shape,
        #                                      batch_size=self.batch_size)
        # else:
        #     return keras.layers.AlphaDropout(_rate,
        #                                      seed=_seed)

    def layer_select(self, select, _config=None):
        if select == 1:
            return self.r_dense(_config=_config)
        elif select == 2:
            return self.r_activation(_config=_config)
        elif select == 3:
            return self.r_dropout(_config=_config)
        elif select == 4:
            return self.r_flatten(_config=_config)
        elif select == 5:
            return self.r_activity_regularization(_config=_config)
        elif select == 6:
            return self.r_masking(_config=_config)
        elif select == 7:
            return self.r_spatial_dropout1d(_config=_config)
        elif select == 8:
            return self.r_spatial_dropout2d(_config=_config)
        elif select == 9:
            return self.r_spatial_dropout3d(_config=_config)
        elif select == 10:
            return self.r_convolution1d(_config=_config)
        elif select == 11:
            return self.r_convolution2d(_config=_config)
        elif select == 12:
            return self.r_separable_conv1d(_config=_config)
        elif select == 13:
            return self.r_separable_conv2d(_config=_config)
        elif select == 14:
            return self.r_depthwise_conv2d(_config=_config)
        elif select == 15:
            return self.r_conv2d_transpose(_config=_config)
        elif select == 16:
            return self.r_convolution3d(_config=_config)
        elif select == 17:
            return self.r_conv3d_transpose(_config=_config)
        elif select == 18:
            return self.r_cropping1d(_config=_config)
        elif select == 19:
            return self.r_cropping2d(_config=_config)
        elif select == 20:
            return self.r_cropping3d(_config=_config)
        elif select == 21:
            return self.r_up_sampling1d(_config=_config)
        elif select == 22:
            return self.r_up_sampling2d(_config=_config)
        elif select == 23:
            return self.r_up_sampling3d(_config=_config)
        elif select == 24:
            return self.r_zero_padding1d(_config=_config)
        elif select == 25:
            return self.r_zero_padding2d(_config=_config)
        elif select == 26:
            return self.r_zero_padding3d(_config=_config)
        elif select == 27:
            return self.r_max_pooling1d(_config=_config)
        elif select == 28:
            return self.r_max_pooling2d(_config=_config)
        elif select == 29:
            return self.r_max_pooling3d(_config=_config)
        elif select == 30:
            return self.r_average_pooling1d(_config=_config)
        elif select == 31:
            return self.r_average_pooling2d(_config=_config)
        elif select == 32:
            return self.r_average_pooling3d(_config=_config)
        elif select == 33:
            return self.r_global_max_pooling1d(_config=_config)
        elif select == 34:
            return self.r_global_average_pooling1d(_config=_config)
        elif select == 35:
            return self.r_global_max_pooling2d(_config=_config)
        elif select == 36:
            return self.r_global_average_pooling2d(_config=_config)
        elif select == 37:
            return self.r_global_max_pooling3d(_config=_config)
        elif select == 38:
            return self.r_global_average_pooling3d(_config=_config)
        elif select == 39:
            return self.r_locally_connected1d(_config=_config)
        elif select == 40:
            return self.r_locally_connected2d(_config=_config)
        elif select == 41:
            return self.r_simple_rnn(_config=_config)
        elif select == 42:
            return self.r_gru(_config=_config)
        elif select == 43:
            return self.r_lstm(_config=_config)
        elif select == 44:
            return self.r_conv_lstm2d(_config=_config)
        elif select == 45:
            return self.r_leaky_relu(_config=_config)
        elif select == 46:
            return self.r_p_relu(_config=_config)
        elif select == 47:
            return self.r_elu(_config=_config)
        elif select == 48:
            return self.r_thresholded_relu(_config=_config)
        elif select == 49:
            return self.r_softmax(_config=_config)
        elif select == 50:
            return self.r_relu(_config=_config)
        elif select == 51:
            return self.r_gaussian_noise(_config=_config)
        elif select == 52:
            return self.r_gaussian_dropout(_config=_config)
        elif select == 53:
            return self.r_alpha_dropout(_config=_config)
        elif select == 54:
            return self.r_embedding(_config=_config)
        elif select == 55:
            return self.r_input()
        elif select == 56:
            return self.r_dense_without_activation(10, _config=_config)
