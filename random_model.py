from random_layer import RandomLayers
from keras.models import Sequential
from utils import *
import random


class RandomModel:
    def __init__(self):
        self.max_layer = 20

    def generate_model(self, input_shape):
        layer_num = random.randint(1, self.max_layer)
        layer_count = 0
        layer_generator = RandomLayers()
        # layer list
        general_layer_list = list(layer_generator.general_layer_map.keys())
        layer_0d_list = list(layer_generator.layer_map_0d.keys())
        layer_1d_list = list(layer_generator.layer_map_1d.keys())
        layer_2d_list = list(layer_generator.layer_map_2d.keys())
        layer_3d_list = list(layer_generator.layer_map_3d.keys())
        layer_rnn_list = list(layer_generator.layer_map_rnn.keys())
        layer_pooling_list = list(layer_generator.layer_map_pooling.keys())

        model = Sequential()
        # input layer
        layer_generator.set_input_shape(input_shape)
        if len(input_shape) == 1:
            layer_generator.now_select_layer = '0d'
            input_layer = layer_generator.r_embedding()
            model.add(input_layer)
        elif len(input_shape) == 2:
            layer_generator.now_select_layer = '1d'
            input_layer = layer_generator.r_input()
            model.add(input_layer)
        elif len(input_shape) == 3:
            layer_generator.now_select_layer = '2d'
            input_layer = layer_generator.r_input()
            model.add(input_layer)
        else:
            layer_generator.now_select_layer = '3d'
            input_layer = layer_generator.r_input()
            model.add(input_layer)
        layer_count += 1
        # remain layers
        while layer_count < layer_num:
            if layer_generator.now_select_layer == '0d':
                if layer_count == layer_num - 1:
                    selected_layer = random.choice(general_layer_list)
                else:
                    selected_layer = random.choice(general_layer_list + layer_rnn_list)
                print(selected_layer)
                if selected_layer in layer_rnn_list:
                    layer_generator.now_select_layer = '-d'
                    select_num = layer_generator.layer_map_rnn[selected_layer]
                    layer = layer_generator.layer_select(select_num)
                    model.add(layer)
            elif layer_generator.now_select_layer == '1d':
                selected_layer = random.choice(layer_0d_list + layer_1d_list + general_layer_list)
                print(selected_layer)
                if selected_layer in layer_0d_list + layer_pooling_list:
                    layer_generator.now_select_layer = '-d'

                if selected_layer in layer_1d_list:
                    select_num = layer_generator.layer_map_1d[selected_layer]
                    layer = layer_generator.layer_select(select_num)
                    model.add(layer)
            elif layer_generator.now_select_layer == '2d':
                selected_layer = random.choice(layer_0d_list + layer_2d_list + general_layer_list)
                print(selected_layer)
                if selected_layer in layer_0d_list + layer_pooling_list:
                    layer_generator.now_select_layer = '-d'
                if selected_layer in layer_2d_list:
                    select_num = layer_generator.layer_map_2d[selected_layer]
                    layer = layer_generator.layer_select(select_num)
                    model.add(layer)
            elif layer_generator.now_select_layer == '-d':
                selected_layer = random.choice(general_layer_list)
                print(selected_layer)

            else:
                selected_layer = random.choice(layer_0d_list + layer_3d_list + general_layer_list)
                print(selected_layer)
                if selected_layer in layer_0d_list + layer_pooling_list:
                    layer_generator.now_select_layer = '-d'
                if selected_layer in layer_3d_list:
                    select_num = layer_generator.layer_map_3d[selected_layer]
                    layer = layer_generator.layer_select(select_num)
                    model.add(layer)
            if selected_layer in general_layer_list:
                select_num = layer_generator.general_layer_map[selected_layer]
                layer = layer_generator.layer_select(select_num)
                model.add(layer)
            layer_count += 1

        _loss = r_loss()
        _optimizer = r_optimizer()
        model.compile(loss=_loss,
                      optimizer=_optimizer,
                      metrics=['accuracy'])
        model.summary()


if __name__ == '__main__':
    test = RandomModel()
    test.generate_model((28, 23, 28))
