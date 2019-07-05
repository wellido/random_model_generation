from utils import *
import random
import keras


class RandomLayers:
    def __init__(self, is_first_layer, input_shape=None):
        self.max_units = 50
        self.max_filter = 50
        self.max_dropout_rate = 1.0
        self.max_regularization_factor = 0.01
        self.is_first_layer = is_first_layer
        self.input_shape = input_shape

    # Dense layer
    def r_dense(self):
        _activation = r_activation()
        _kernel_initializer = r_initializer()
        _bias_initializer = r_initializer()
        _kernel_regularizer = r_regularizers()
        _bias_regularizer = r_regularizers()
        _unit = random.randint(1, self.max_units)
        if self.is_first_layer:
            return keras.layers.Dense(_unit, activation=_activation, kernel_initializer=_kernel_initializer,
                                      bias_initializer=_bias_initializer, kernel_regularizer=_kernel_regularizer,
                                      bias_regularizer=_bias_regularizer)
        else:
            return keras.layers.Dense(_unit, activation=_activation, kernel_initializer=_kernel_initializer,
                                      bias_initializer=_bias_initializer, kernel_regularizer=_kernel_regularizer,
                                      bias_regularizer=_bias_regularizer, input_shape=self.input_shape)

    # Activation layer
    def r_activation(self):
        _activation = r_activation()
        if self.is_first_layer:
            return keras.layers.Activation(_activation)
        else:
            return keras.layers.Activation(_activation, input_shape=self.input_shape)

    # Dropout layer
    def r_dropout(self):
        dropout_rate = random.uniform(0., self.max_dropout_rate)
        _seed = random.sample([None, 1, 2, 3, 4, 5], 1)[0]
        if self.is_first_layer:
            return keras.layers.Dropout(dropout_rate, seed=_seed)
        else:
            return keras.layers.Dropout(dropout_rate, seed=_seed, input_shape=self.input_shape)

    # Flatten layer
    def r_flatten(self):
        if self.is_first_layer:
            return keras.layers.Flatten()
        else:
            return keras.layers.Flatten(input_shape=self.input_shape)

    # Input layer
    def r_input(self):
        return keras.engine.input_layer.Input(shape=self.input_shape)

    # ActivityRegularization layer
    def r_activity_regularization(self):
        l1_factor = random.uniform(0., self.max_regularization_factor)
        l2_factor = random.uniform(0., self.max_regularization_factor)
        if self.is_first_layer:
            return keras.layers.ActivityRegularization(l1=l1_factor, l2=l2_factor)
        else:
            return keras.layers.ActivityRegularization(l1=l1_factor, l2=l2_factor, input_shape=self.input_shape)

    # Masking layer
    def r_masking(self):
        _mask_value = random.uniform(0., 1.0)
        return keras.layers.Masking(mask_value=_mask_value, input_shape=self.input_shape)

    # SpatialDropout1D layer
    def r_spatial_dropout1d(self):
        dropout_rate = random.uniform(0., self.max_dropout_rate)
        if self.is_first_layer:
            return keras.layers.SpatialDropout1D(dropout_rate)
        else:
            return keras.layers.SpatialDropout1D(dropout_rate, input_shape=self.input_shape)

    # SpatialDropout2D layer
    def r_spatial_dropout2d(self):
        dropout_rate = random.uniform(0., self.max_dropout_rate)
        if self.is_first_layer:
            return keras.layers.SpatialDropout2D(dropout_rate)
        else:
            return keras.layers.SpatialDropout2D(dropout_rate, input_shape=self.input_shape)

    # SpatialDropout3D layer
    def r_spatial_dropout3d(self):
        dropout_rate = random.uniform(0., self.max_dropout_rate)
        if self.is_first_layer:
            return keras.layers.SpatialDropout3D(dropout_rate)
        else:
            return keras.layers.SpatialDropout3D(dropout_rate, input_shape=self.input_shape)

    # Conv1D layer
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
            return keras.layers.Conv1D(_filters, _kernel_size, _strides, padding=_padding, activation=_activation,
                                       kernel_initializer=_kernel_initializer, bias_initializer=_bias_initializer,
                                       kernel_regularizer=_kernel_regularizer, bias_regularizer=_bias_regularizer,
                                       activity_regularizer=_activity_regularizer)
        else:
            return keras.layers.Conv1D(_filters, _kernel_size, _strides, padding=_padding, activation=_activation,
                                       kernel_initializer=_kernel_initializer, bias_initializer=_bias_initializer,
                                       kernel_regularizer=_kernel_regularizer, bias_regularizer=_bias_regularizer,
                                       activity_regularizer=_activity_regularizer, input_shape=self.input_shape)

    # Conv2D layer
    def r_convolution2d(self):
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
            return keras.layers.Conv2D(_filters, (_kernel_size, _kernel_size), (_strides, _strides),
                                       padding=_padding, activation=_activation,
                                       kernel_initializer=_kernel_initializer, bias_initializer=_bias_initializer,
                                       kernel_regularizer=_kernel_regularizer, bias_regularizer=_bias_regularizer,
                                       activity_regularizer=_activity_regularizer)
        else:
            return keras.layers.Conv2D(_filters, (_kernel_size, _kernel_size), (_strides, _strides),
                                       padding=_padding, activation=_activation,
                                       kernel_initializer=_kernel_initializer, bias_initializer=_bias_initializer,
                                       kernel_regularizer=_kernel_regularizer, bias_regularizer=_bias_regularizer,
                                       activity_regularizer=_activity_regularizer, input_shape=self.input_shape)

    # Conv3D layer
    def r_convolution3d(self):
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
            return keras.layers.Conv2D(_filters, (_kernel_size, _kernel_size, _kernel_size),
                                       (_strides, _strides, _strides),
                                       padding=_padding, activation=_activation,
                                       kernel_initializer=_kernel_initializer, bias_initializer=_bias_initializer,
                                       kernel_regularizer=_kernel_regularizer, bias_regularizer=_bias_regularizer,
                                       activity_regularizer=_activity_regularizer)
        else:
            return keras.layers.Conv2D(_filters, (_kernel_size, _kernel_size, _kernel_size),
                                       (_strides, _strides, _strides),
                                       padding=_padding, activation=_activation,
                                       kernel_initializer=_kernel_initializer, bias_initializer=_bias_initializer,
                                       kernel_regularizer=_kernel_regularizer, bias_regularizer=_bias_regularizer,
                                       activity_regularizer=_activity_regularizer, input_shape=self.input_shape)

    # SeparableConv1D layer
    def r_separableconv1d(self):
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
            return keras.layers.SeparableConv1D(_filters, _kernel_size, _strides,
                                                padding=_padding, activation=_activation,
                                                kernel_initializer=_kernel_initializer,
                                                bias_initializer=_bias_initializer,
                                                kernel_regularizer=_kernel_regularizer,
                                                bias_regularizer=_bias_regularizer,
                                                activity_regularizer=_activity_regularizer)
        else:
            return keras.layers.SeparableConv1D(_filters, _kernel_size, _strides,
                                                padding=_padding, activation=_activation,
                                                kernel_initializer=_kernel_initializer,
                                                bias_initializer=_bias_initializer,
                                                kernel_regularizer=_kernel_regularizer,
                                                bias_regularizer=_bias_regularizer,
                                                activity_regularizer=_activity_regularizer,
                                                input_shape=self.input_shape)

    # SeparableConv2D layer
    def r_separableconv2d(self):
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
            return keras.layers.SpatialDropout2D(_filters, (_kernel_size, _kernel_size), (_strides, _strides),
                                                 padding=_padding, activation=_activation,
                                                 kernel_initializer=_kernel_initializer,
                                                 bias_initializer=_bias_initializer,
                                                 kernel_regularizer=_kernel_regularizer,
                                                 bias_regularizer=_bias_regularizer,
                                                 activity_regularizer=_activity_regularizer)
        else:
            return keras.layers.SeparableConv2D(_filters, (_kernel_size, _kernel_size), (_strides, _strides),
                                                padding=_padding, activation=_activation,
                                                kernel_initializer=_kernel_initializer,
                                                bias_initializer=_bias_initializer,
                                                kernel_regularizer=_kernel_regularizer,
                                                bias_regularizer=_bias_regularizer,
                                                activity_regularizer=_activity_regularizer,
                                                input_shape=self.input_shape)

    # DepthwiseConv2D layer
    def r_depthwiseconv2d(self):
        _kernel_size = random.randint(1, self.max_filter)
        _strides = random.randint(1, self.max_filter - _kernel_size + 1)
        _padding = r_padding()
        _activation = r_activation()
        _kernel_initializer = r_initializer()
        _bias_initializer = r_initializer()
        _kernel_regularizer = r_regularizers()
        _bias_regularizer = r_regularizers()
        _activity_regularizer = r_regularizers()
        if self.is_first_layer:
            return keras.layers.SpatialDropout2D((_kernel_size, _kernel_size), (_strides, _strides),
                                                 padding=_padding, activation=_activation,
                                                 kernel_initializer=_kernel_initializer,
                                                 bias_initializer=_bias_initializer,
                                                 kernel_regularizer=_kernel_regularizer,
                                                 bias_regularizer=_bias_regularizer,
                                                 activity_regularizer=_activity_regularizer)
        else:
            return keras.layers.SeparableConv2D((_kernel_size, _kernel_size), (_strides, _strides),
                                                padding=_padding, activation=_activation,
                                                kernel_initializer=_kernel_initializer,
                                                bias_initializer=_bias_initializer,
                                                kernel_regularizer=_kernel_regularizer,
                                                bias_regularizer=_bias_regularizer,
                                                activity_regularizer=_activity_regularizer,
                                                input_shape=self.input_shape)

    # Conv2DTranspose layer
    
