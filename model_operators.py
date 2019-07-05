from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
import keras


def layer_remove(model, layer_name):
    """
    remove one layer
    selected layer must has the same input and output shape
    :param model: original model
    :param layer_name: layer selected
    :return: mutant model
    """
    new_model = Sequential()
    for layer in model.layers:
        if layer.name == layer_name:
            continue
        new_model.add(layer)
    return new_model


# model = load_model("models/lenet5.h5")
# model.summary()
# new_model = layer_remove(model, "conv2d_2")
# new_model.summary()
test_model = Sequential()
# test_model.add(keras.layers.RepeatVector(2, input_shape=(2,)))
# test_model.add(MaxPooling2D(pool_size=(2, 2), input_shape=(4, 4, 4)))
# test_model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), input_shape=(28, 28, 1)))
test_model.add(keras.layers.Embedding(20000, 256))
test_model.add(keras.layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2, input_shape=(1, 128)))
# test_model.add(Conv2D(32, kernel_size=(3, 3),
#                  activation='relu',
#                  input_shape=(28, 28, 1)))

# test_model.add(Flatten())
# test_model.add(Activation('relu'))
# test_model.add(Dropout(0.25))
test_model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

test_model.summary()
