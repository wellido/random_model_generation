from utils import *
import random
import keras


class RandomLayers:
    def __init__(self):
        self.max_units = 50
        self.max_filter = 50
        self.max_cell = 50
        self.max_dropout_rate = 1.0
        self.max_regularization_factor = 0.01
        self.max_input_dim = 2000
        self.max_output_dim = 100
        self.units = 0
        self.input_shape = None
        self.is_first_layer = 0
        self.now_select_layer = 'all'

    def set_first_layer(self, is_first_layer):
        self.is_first_layer = is_first_layer

    def set_input_shape(self, input_shape):
        self.input_shape = input_shape

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
    def r_dense(self):
        _activation = r_activation()
        _kernel_initializer = r_initializer()
        _bias_initializer = r_initializer()
        _kernel_regularizer = r_regularizers()
        _bias_regularizer = r_regularizers()
        _unit = random.randint(1, self.max_units)
        self.units = _unit
        if self.is_first_layer:
            return keras.layers.Dense(_unit,
                                      activation=_activation,
                                      kernel_initializer=_kernel_initializer,
                                      bias_initializer=_bias_initializer,
                                      kernel_regularizer=_kernel_regularizer,
                                      bias_regularizer=_bias_regularizer,
                                      input_shape=self.input_shape)
        else:
            return keras.layers.Dense(_unit,
                                      activation=_activation,
                                      kernel_initializer=_kernel_initializer,
                                      bias_initializer=_bias_initializer,
                                      kernel_regularizer=_kernel_regularizer,
                                      bias_regularizer=_bias_regularizer)

    # Activation Layer
    def r_activation(self):
        _activation = r_activation()
        if self.is_first_layer:
            return keras.layers.Activation(_activation,
                                           input_shape=self.input_shape)
        else:
            return keras.layers.Activation(_activation)

    # Dropout Layer
    def r_dropout(self):
        dropout_rate = random.uniform(0., self.max_dropout_rate)
        # _seed = random.sample([None, 1, 2, 3, 4, 5], 1)[0]
        _seed = random.choice([None, 1, 2, 3, 4, 5])
        if self.is_first_layer:
            return keras.layers.Dropout(dropout_rate,
                                        seed=_seed,
                                        input_shape=self.input_shape)
        else:
            return keras.layers.Dropout(dropout_rate,
                                        seed=_seed)

    # Flatten Layer
    def r_flatten(self):
        if self.is_first_layer:
            return keras.layers.Flatten(input_shape=self.input_shape)
        else:
            return keras.layers.Flatten()

    # Input Layer
    def r_input(self):
        return keras.layers.InputLayer(input_shape=self.input_shape, dtype='float32')

    # ActivityRegularization Layer
    def r_activity_regularization(self):
        l1_factor = random.uniform(0., self.max_regularization_factor)
        l2_factor = random.uniform(0., self.max_regularization_factor)
        if self.is_first_layer:
            return keras.layers.ActivityRegularization(l1=l1_factor,
                                                       l2=l2_factor,
                                                       input_shape=self.input_shape)
        else:
            return keras.layers.ActivityRegularization(l1=l1_factor,
                                                       l2=l2_factor)

    # Masking Layer
    def r_masking(self):
        _mask_value = random.uniform(0., 1.0)
        return keras.layers.Masking(mask_value=_mask_value,
                                    input_shape=self.input_shape)

    # SpatialDropout1D Layer
    def r_spatial_dropout1d(self):
        dropout_rate = random.uniform(0., self.max_dropout_rate)
        if self.is_first_layer:
            return keras.layers.SpatialDropout1D(dropout_rate)
        else:
            return keras.layers.SpatialDropout1D(dropout_rate,
                                                 input_shape=self.input_shape)

    # SpatialDropout2D Layer
    def r_spatial_dropout2d(self):
        dropout_rate = random.uniform(0., self.max_dropout_rate)
        if self.is_first_layer:
            return keras.layers.SpatialDropout2D(dropout_rate,
                                                 input_shape=self.input_shape)
        else:
            return keras.layers.SpatialDropout2D(dropout_rate)

    # SpatialDropout3D Layer
    def r_spatial_dropout3d(self):
        dropout_rate = random.uniform(0., self.max_dropout_rate)
        if self.is_first_layer:
            return keras.layers.SpatialDropout3D(dropout_rate,
                                                 input_shape=self.input_shape)
        else:
            return keras.layers.SpatialDropout3D(dropout_rate)

    # Conv1D Layer
    def r_convolution1d(self):
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
        if self.is_first_layer:
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
                                       input_shape=self.input_shape)
        else:
            return keras.layers.Conv1D(_filters,
                                       _kernel_size,
                                       strides=_strides,
                                       padding=_padding,
                                       activation=_activation,
                                       kernel_initializer=_kernel_initializer,
                                       bias_initializer=_bias_initializer,
                                       kernel_regularizer=_kernel_regularizer,
                                       bias_regularizer=_bias_regularizer,
                                       activity_regularizer=_activity_regularizer)

    # Conv2D Layer
    def r_convolution2d(self):
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
        if self.is_first_layer:
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
                                       input_shape=self.input_shape)
        else:
            return keras.layers.Conv2D(_filters,
                                       (_kernel_size, _kernel_size),
                                       strides=(_strides, _strides),
                                       padding=_padding,
                                       activation=_activation,
                                       kernel_initializer=_kernel_initializer,
                                       bias_initializer=_bias_initializer,
                                       kernel_regularizer=_kernel_regularizer,
                                       bias_regularizer=_bias_regularizer,
                                       activity_regularizer=_activity_regularizer)

    # SeparableConv1D Layer
    def r_separable_conv1d(self):
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
        if self.is_first_layer:
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
                                                input_shape=self.input_shape)
        else:
            return keras.layers.SeparableConv1D(_filters,
                                                _kernel_size,
                                                _strides,
                                                padding=_padding,
                                                activation=_activation,
                                                kernel_initializer=_kernel_initializer,
                                                bias_initializer=_bias_initializer,
                                                kernel_regularizer=_kernel_regularizer,
                                                bias_regularizer=_bias_regularizer,
                                                activity_regularizer=_activity_regularizer)

    # SeparableConv2D Layer
    def r_separable_conv2d(self):
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
        if self.is_first_layer:
            return keras.layers.SpatialDropout2D(_filters,
                                                 (_kernel_size, _kernel_size),
                                                 strides=(_strides, _strides),
                                                 padding=_padding,
                                                 activation=_activation,
                                                 kernel_initializer=_kernel_initializer,
                                                 bias_initializer=_bias_initializer,
                                                 kernel_regularizer=_kernel_regularizer,
                                                 bias_regularizer=_bias_regularizer,
                                                 activity_regularizer=_activity_regularizer,
                                                 input_shape=self.input_shape)
        else:
            return keras.layers.SeparableConv2D(_filters,
                                                (_kernel_size, _kernel_size),
                                                strides=(_strides, _strides),
                                                padding=_padding,
                                                activation=_activation,
                                                kernel_initializer=_kernel_initializer,
                                                bias_initializer=_bias_initializer,
                                                kernel_regularizer=_kernel_regularizer,
                                                bias_regularizer=_bias_regularizer,
                                                activity_regularizer=_activity_regularizer)

    # DepthwiseConv2D Layer
    def r_depthwise_conv2d(self):
        _kernel_size = random.randint(1, self.max_filter)
        _strides = random.randint(1, self.max_filter - _kernel_size + 1)
        _padding = r_padding2()
        _activation = r_activation()
        _kernel_initializer = r_initializer()
        _bias_initializer = r_initializer()
        _kernel_regularizer = r_regularizers()
        _bias_regularizer = r_regularizers()
        _activity_regularizer = r_regularizers()
        if self.is_first_layer:
            return keras.layers.DepthwiseConv2D(_kernel_size,
                                                strides=(_strides, _strides),
                                                padding=_padding,
                                                activation=_activation,
                                                kernel_initializer=_kernel_initializer,
                                                bias_initializer=_bias_initializer,
                                                kernel_regularizer=_kernel_regularizer,
                                                bias_regularizer=_bias_regularizer,
                                                activity_regularizer=_activity_regularizer,
                                                input_shape=self.input_shape)
        else:
            return keras.layers.DepthwiseConv2D(_kernel_size,
                                                strides=(_strides, _strides),
                                                padding=_padding,
                                                activation=_activation,
                                                kernel_initializer=_kernel_initializer,
                                                bias_initializer=_bias_initializer,
                                                kernel_regularizer=_kernel_regularizer,
                                                bias_regularizer=_bias_regularizer,
                                                activity_regularizer=_activity_regularizer)

    # Conv2DTranspose Layer
    def r_conv2d_transpose(self):
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
        if self.is_first_layer:
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
                                                input_shape=self.input_shape
                                                )
        else:
            return keras.layers.Conv2DTranspose(_filters,
                                                (_kernel_size, _kernel_size),
                                                strides=(_strides, _strides),
                                                padding=_padding,
                                                activation=_activation,
                                                kernel_initializer='glorot_uniform',
                                                bias_initializer='zeros',
                                                kernel_regularizer=_kernel_regularizer,
                                                bias_regularizer=_bias_regularizer,
                                                activity_regularizer=_activity_regularizer
                                                )

    # Conv3D Layer
    def r_convolution3d(self):
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
        if self.is_first_layer:
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
                                       input_shape=self.input_shape)
        else:
            return keras.layers.Conv3D(_filters,
                                       (_kernel_size, _kernel_size, _kernel_size),
                                       strides=(_strides, _strides, _strides),
                                       padding=_padding,
                                       activation=_activation,
                                       kernel_initializer=_kernel_initializer,
                                       bias_initializer=_bias_initializer,
                                       kernel_regularizer=_kernel_regularizer,
                                       bias_regularizer=_bias_regularizer,
                                       activity_regularizer=_activity_regularizer)

    # Conv3DTranspose Layer
    def r_conv3d_transpose(self):
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
        if self.is_first_layer:
            return keras.layers.Conv3DTranspose(_filters,
                                                (_kernel_size, _kernel_size, _kernel_size),
                                                strides=(_strides, _strides, _strides),
                                                padding=_padding, activation=_activation,
                                                kernel_initializer=_kernel_initializer,
                                                bias_initializer=_bias_initializer,
                                                kernel_regularizer=_kernel_regularizer,
                                                bias_regularizer=_bias_regularizer,
                                                activity_regularizer=_activity_regularizer
                                                )
        else:
            return keras.layers.Conv3DTranspose(_filters,
                                                (_kernel_size, _kernel_size, _kernel_size),
                                                strides=(_strides, _strides, _strides),
                                                padding=_padding, activation=_activation,
                                                kernel_initializer=_kernel_initializer,
                                                bias_initializer=_bias_initializer,
                                                kernel_regularizer=_kernel_regularizer,
                                                bias_regularizer=_bias_regularizer,
                                                activity_regularizer=_activity_regularizer,
                                                input_shape=self.input_shape
                                                )

    # Cropping1D Layer
    def r_cropping1d(self):
        _cropping = random.randint(0, 2)
        if self.is_first_layer:
            return keras.layers.Cropping1D(cropping=_cropping,
                                           input_shape=self.input_shape)
        else:
            return keras.layers.Cropping1D(cropping=_cropping)

    # Cropping2D Layer
    def r_cropping2d(self):
        _cropping = random.randint(0, 2)
        if self.is_first_layer:
            return keras.layers.Cropping2D(cropping=_cropping,
                                           input_shape=self.input_shape)
        else:
            return keras.layers.Cropping2D(cropping=_cropping)

    # Cropping3D Layer
    def r_cropping3d(self):
        _cropping = random.randint(0, 2)
        if self.is_first_layer:
            return keras.layers.Cropping3D(cropping=_cropping,
                                           input_shape=self.input_shape)
        else:
            return keras.layers.Cropping3D(cropping=_cropping)

    # UpSampling1D Layer
    def r_up_sampling1d(self):
        _size = random.randint(0, 2)
        if self.is_first_layer:
            return keras.layers.UpSampling1D(size=_size,
                                             input_shape=self.input_shape)
        else:
            return keras.layers.UpSampling1D(size=_size)

    # UpSampling2D Layer
    def r_up_sampling2d(self):
        _size = random.randint(0, 2)
        _interpolation = r_interpolation()
        if self.is_first_layer:
            return keras.layers.UpSampling2D(size=_size,
                                             interpolation=_interpolation,
                                             input_shape=self.input_shape)
        else:
            return keras.layers.UpSampling2D(size=_size,
                                             interpolation=_interpolation)

    # UpSampling3D Layer
    def r_up_sampling3d(self):
        _size = random.randint(0, 2)
        if self.is_first_layer:
            return keras.layers.UpSampling3D(size=_size,
                                             input_shape=self.input_shape)
        else:
            return keras.layers.UpSampling3D(size=_size)

    # ZeroPadding1D Layer
    def r_zero_padding1d(self):
        _choose = random.randint(0, 1)
        _padding = random.randint(0, 9)
        if _choose:
            if self.is_first_layer:
                return keras.layers.ZeroPadding1D(padding=(_padding, _padding),
                                                  input_shape=self.input_shape)
            else:
                return keras.layers.ZeroPadding1D(padding=(_padding, _padding))
        else:
            return keras.layers.ZeroPadding1D(padding=_padding)

    # ZeroPadding2D Layer
    def r_zero_padding2d(self):
        _padding = random.randint(0, 9)
        if self.is_first_layer:
            return keras.layers.ZeroPadding2D(padding=_padding,
                                              data_format=None,
                                              input_shape=self.input_shape)
        else:
            return keras.layers.ZeroPadding2D(padding=_padding,
                                              data_format=None)

    # ZeroPadding3D Layer
    def r_zero_padding3d(self):
        _padding = random.randint(0, 9)
        if self.is_first_layer:
            return keras.layers.ZeroPadding3D(padding=_padding,
                                              data_format=None,
                                              input_shape=self.input_shape)
        else:
            return keras.layers.ZeroPadding3D(padding=_padding,
                                              data_format=None)

    # MaxPooling1D Layer
    def r_max_pooling1d(self):
        _pool_size = random.randint(1, 3)
        _padding = r_padding2()
        if self.is_first_layer:
            return keras.layers.MaxPooling1D(pool_size=_pool_size,
                                             padding=_padding,
                                             input_shape=self.input_shape)
        else:
            return keras.layers.MaxPooling1D(pool_size=_pool_size,
                                             padding=_padding)

    # MaxPooling2D Layer
    def r_max_pooling2d(self):
        _pool_size = random.randint(1, 3)
        _padding = r_padding2()
        if self.is_first_layer:
            return keras.layers.MaxPooling2D(pool_size=_pool_size,
                                             padding=_padding,
                                             input_shape=self.input_shape)
        else:
            return keras.layers.MaxPooling2D(pool_size=_pool_size,
                                             padding=_padding)

    # MaxPooling3D Layer
    def r_max_pooling3d(self):
        _pool_size = random.randint(1, 3)
        _padding = r_padding2()
        if self.is_first_layer:
            return keras.layers.MaxPooling3D(pool_size=_pool_size,
                                             padding=_padding,
                                             input_shape=self.input_shape)
        else:
            return keras.layers.MaxPooling3D(pool_size=_pool_size,
                                             padding=_padding)

    # AveragePooling1D Layer
    def r_average_pooling1d(self):
        _pool_size = random.randint(1, 3)
        _padding = r_padding2()
        if self.is_first_layer:
            return keras.layers.AveragePooling1D(pool_size=_pool_size,
                                                 padding=_padding,
                                                 input_shape=self.input_shape)
        else:
            return keras.layers.AveragePooling1D(pool_size=_pool_size,
                                                 padding=_padding)

    # AveragePooling2D Layer
    def r_average_pooling2d(self):
        _pool_size = random.randint(1, 3)
        _padding = r_padding2()
        if self.is_first_layer:
            return keras.layers.AveragePooling2D(pool_size=_pool_size,
                                                 padding=_padding,
                                                 input_shape=self.input_shape)
        else:
            return keras.layers.AveragePooling2D(pool_size=_pool_size,
                                                 padding=_padding)

    # AveragePooling3D Layer
    def r_average_pooling3d(self):
        _pool_size = random.randint(1, 3)
        _padding = r_padding2()
        if self.is_first_layer:
            return keras.layers.AveragePooling3D(pool_size=_pool_size,
                                                 padding=_padding,
                                                 input_shape=self.input_shape)
        else:
            return keras.layers.AveragePooling3D(pool_size=_pool_size,
                                                 padding=_padding)

    # GlobalMaxPooling1D Layer
    def r_global_max_pooling1d(self):
        if self.is_first_layer:
            return keras.layers.GlobalMaxPooling1D(input_shape=self.input_shape)
        else:
            return keras.layers.GlobalAveragePooling1D()

    # GlobalAveragePooling1D Layer
    def r_global_average_pooling1d(self):
        if self.is_first_layer:
            return keras.layers.GlobalAveragePooling1D(input_shape=self.input_shape)
        else:
            return keras.layers.GlobalAveragePooling1D()

    # GlobalMaxPooling2D Layer
    def r_global_max_pooling2d(self):
        if self.is_first_layer:
            return keras.layers.GlobalMaxPooling2D(input_shape=self.input_shape)
        else:
            return keras.layers.GlobalMaxPooling2D()

    # GlobalAveragePooling2D Layer
    def r_global_average_pooling2d(self):
        if self.is_first_layer:
            return keras.layers.GlobalAveragePooling2D(input_shape=self.input_shape)
        else:
            return keras.layers.GlobalAveragePooling2D()

    # GlobalMaxPooling3D Layer
    def r_global_max_pooling3d(self):
        if self.is_first_layer:
            return keras.layers.GlobalMaxPooling3D(input_shape=self.input_shape)
        else:
            return keras.layers.GlobalMaxPooling3D()

    # GlobalAveragePooling3D Layer
    def r_global_average_pooling3d(self):
        if self.is_first_layer:
            return keras.layers.GlobalAveragePooling3D(input_shape=self.input_shape)
        else:
            return keras.layers.GlobalAveragePooling3D()

    # LocallyConnected1D Layer
    def r_locally_connected1d(self):
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
        if self.is_first_layer:
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
                                                   input_shape=self.input_shape)
        else:
            return keras.layers.LocallyConnected1D(_filters,
                                                   _kernel_size,
                                                   _strides,
                                                   padding=_padding,
                                                   activation=_activation,
                                                   kernel_initializer=_kernel_initializer,
                                                   bias_initializer=_bias_initializer,
                                                   kernel_regularizer=_kernel_regularizer,
                                                   bias_regularizer=_bias_regularizer,
                                                   activity_regularizer=_activity_regularizer)

    # LocallyConnected2D Layer
    def r_locally_connected2d(self):
        _filters = random.randint(1, self.max_filter)
        _kernel_size = random.randint(1, _filters)
        _strides = random.randint(1, _filters - _kernel_size + 1)
        _activation = r_activation()
        _kernel_initializer = r_initializer()
        _bias_initializer = r_initializer()
        _kernel_regularizer = r_regularizers()
        _bias_regularizer = r_regularizers()
        _activity_regularizer = r_regularizers()
        if self.is_first_layer:
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
                                                   input_shape=self.input_shape)
        else:
            return keras.layers.LocallyConnected2D(_filters,
                                                   (_kernel_size, _kernel_size),
                                                   strides=(_strides, _strides),
                                                   padding='valid',
                                                   activation=_activation,
                                                   kernel_initializer=_kernel_initializer,
                                                   bias_initializer=_bias_initializer,
                                                   kernel_regularizer=_kernel_regularizer,
                                                   bias_regularizer=_bias_regularizer,
                                                   activity_regularizer=_activity_regularizer)

    # SimpleRNN Layer
    def r_simple_rnn(self):
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
        if self.is_first_layer:
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
                                          input_shape=self.input_shape)
        else:
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
                                          unroll=_unroll)

    # GRU Layer
    def r_gru(self):
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
        if self.is_first_layer:
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
                                    input_shape=self.input_shape
                                    )
        else:
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
                                    unroll=_unroll
                                    )

    # LSTM Layer
    def r_lstm(self):
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
        if self.is_first_layer:
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
                                     input_shape=self.input_shape
                                     )
        else:
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
                                     unroll=_unroll
                                     )

    # ConvLSTM2D Layer
    def r_conv_lstm2d(self):
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
        if self.is_first_layer:
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
                                           input_shape=self.input_shape)
        else:
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
                                           recurrent_dropout=_recurent_dropout
                                           )

    # Embedding Layer
    def r_embedding(self):
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
                                      input_length=_input_length)

    # LeakyReLU Layer
    def r_leaky_relu(self):
        _alpha = random.uniform(0., 1.0)
        if self.is_first_layer:
            return keras.layers.LeakyReLU(alpha=_alpha,
                                          input_shape=self.input_shape)
        else:
            return keras.layers.LeakyReLU(alpha=_alpha)

    # PReLU Layer
    def r_p_relu(self):
        _alpha_initializer = r_initializer()
        _alpha_regularizer = r_regularizers()
        if self.is_first_layer:
            return keras.layers.PReLU(alpha_initializer=_alpha_initializer,
                                      alpha_regularizer=_alpha_regularizer,
                                      input_shape=self.input_shape)
        else:
            return keras.layers.PReLU(alpha_initializer=_alpha_initializer,
                                      alpha_regularizer=_alpha_regularizer)

    # ELU Layer
    def r_elu(self):
        _alpha = random.uniform(0., 1.0)
        if self.is_first_layer:
            return keras.layers.ELU(alpha=_alpha,
                                    input_shape=self.input_shape)
        else:
            return keras.layers.ELU(alpha=_alpha)

    # ThresholdedReLU Layer
    def r_thresholded_relu(self):
        _theta = random.uniform(0., 1.0)
        if self.is_first_layer:
            return keras.layers.ThresholdedReLU(theta=_theta,
                                                input_shape=self.input_shape)
        else:
            return keras.layers.ThresholdedReLU(theta=_theta)

    # Softmax Layer
    def r_softmax(self):
        if self.is_first_layer:
            return keras.layers.Softmax(input_shape=self.input_shape)
        else:
            return keras.layers.Softmax()

    # ReLU Layer
    def r_relu(self):
        if self.is_first_layer:
            return keras.layers.ReLU(input_shape=self.input_shape)
        else:
            return keras.layers.ReLU()

    # GaussianNoise Layer
    def r_gaussian_noise(self):
        _stddev = random.uniform(0., 1.)
        if self.is_first_layer:
            return keras.layers.GaussianNoise(stddev=_stddev,
                                              input_shape=self.input_shape)
        else:
            return keras.layers.GaussianNoise(stddev=_stddev)

    # GaussianDropout Layer
    def r_gaussian_dropout(self):
        _rate = random.uniform(0., self.max_dropout_rate)
        if self.is_first_layer:
            return keras.layers.GaussianDropout(rate=_rate,
                                                input_shape=self.input_shape)
        else:
            return keras.layers.GaussianDropout(rate=_rate)

    # AlphaDropout Layer
    def r_alpha_dropout(self):
        _rate = random.uniform(0., self.max_dropout_rate)
        _seed = random.randint(1, 10)
        if self.is_first_layer:
            return keras.layers.AlphaDropout(_rate,
                                             seed=_seed,
                                             input_shape=self.input_shape)
        else:
            return keras.layers.AlphaDropout(_rate,
                                             seed=_seed)

    def layer_select(self, select):
        if select == 1:
            return self.r_dense()
        elif select == 2:
            return self.r_activation()
        elif select == 3:
            return self.r_dropout()
        elif select == 4:
            return self.r_flatten()
        elif select == 5:
            return self.r_activity_regularization()
        elif select == 6:
            return self.r_masking()
        elif select == 7:
            return self.r_spatial_dropout1d()
        elif select == 8:
            return self.r_spatial_dropout2d()
        elif select == 9:
            return self.r_spatial_dropout3d()
        elif select == 10:
            return self.r_convolution1d()
        elif select == 11:
            return self.r_convolution2d()
        elif select == 12:
            return self.r_separable_conv1d()
        elif select == 13:
            return self.r_separable_conv2d()
        elif select == 14:
            return self.r_depthwise_conv2d()
        elif select == 15:
            return self.r_conv2d_transpose()
        elif select == 16:
            return self.r_convolution3d()
        elif select == 17:
            return self.r_conv3d_transpose()
        elif select == 18:
            return self.r_cropping1d()
        elif select == 19:
            return self.r_cropping2d()
        elif select == 20:
            return self.r_cropping3d()
        elif select == 21:
            return self.r_up_sampling1d()
        elif select == 22:
            return self.r_up_sampling2d()
        elif select == 23:
            return self.r_up_sampling3d()
        elif select == 24:
            return self.r_zero_padding1d()
        elif select == 25:
            return self.r_zero_padding2d()
        elif select == 26:
            return self.r_zero_padding3d()
        elif select == 27:
            return self.r_max_pooling1d()
        elif select == 28:
            return self.r_max_pooling2d()
        elif select == 29:
            return self.r_max_pooling3d()
        elif select == 30:
            return self.r_average_pooling1d()
        elif select == 31:
            return self.r_average_pooling2d()
        elif select == 32:
            return self.r_average_pooling3d()
        elif select == 33:
            return self.r_global_max_pooling1d()
        elif select == 34:
            return self.r_global_average_pooling1d()
        elif select == 35:
            return self.r_global_max_pooling2d()
        elif select == 36:
            return self.r_global_average_pooling2d()
        elif select == 37:
            return self.r_global_max_pooling3d()
        elif select == 38:
            return self.r_global_average_pooling3d()
        elif select == 39:
            return self.r_locally_connected1d()
        elif select == 40:
            return self.r_locally_connected2d()
        elif select == 41:
            return self.r_simple_rnn()
        elif select == 42:
            return self.r_gru()
        elif select == 43:
            return self.r_lstm()
        elif select == 44:
            return self.r_conv_lstm2d()
        elif select == 45:
            return self.r_leaky_relu()
        elif select == 46:
            return self.r_p_relu()
        elif select == 47:
            return self.r_elu()
        elif select == 48:
            return self.r_thresholded_relu()
        elif select == 49:
            return self.r_softmax()
        elif select == 50:
            return self.r_relu()
        elif select == 51:
            return self.r_gaussian_noise()
        elif select == 52:
            return self.r_gaussian_dropout()
        elif select == 53:
            return self.r_alpha_dropout()
        elif select == 54:
            return self.r_embedding()
        elif select == 55:
            return self.r_input()
