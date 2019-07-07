from random_layer import RandomLayers
from keras.models import Sequential
from utils import *
import random


class RandomModel:
    def __init__(self):
        self.max_layer = 20

    def generate_model(self):
        layer_num = random.randint(1, self.max_layer)
        layer_count = 0
        layer_generator = RandomLayers()
        general_layer_list = list(layer_generator.general_layer_map.keys())
        layer_1d_list = list(layer_generator.layer_map_1d.keys())
        layer_2d_list = list(layer_generator.layer_map_2d.keys())
        layer_3d_list = list(layer_generator.layer_map_3d.keys())
        layer_rnn_list = list(layer_generator.layer_map_rnn.keys())
        all_layer_list = general_layer_list + \
                         layer_1d_list + \
                         layer_2d_list + \
                         layer_3d_list + \
                         layer_rnn_list
        print(all_layer_list)
        model = Sequential()
        layer_generator.set_input_shape((28, 28))
        input_layer = layer_generator.r_input()
        model.add(input_layer)
        layer_count += 1
        print(random.choice(list(layer_generator.general_layer_map.keys())))

        # while layer_count < layer_num:
        #     if layer_count == 1:
        #         layer = layer_generator.layer_select()
        #         model.add(layer)
        #         layer_count += 1
        #     else:
        #         ...
        #     layer_count += 1
        #
        # _loss = r_loss()
        # _optimizer = r_optimizer()
        # model.compile(loss=_loss,
        #               optimizer=_optimizer,
        #               metrics=['accuracy'])
        # model.summary()


if __name__ == '__main__':
    test = RandomModel()
    test.generate_model()
