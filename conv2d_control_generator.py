from random_layer import RandomLayers
from keras.models import Sequential

from utils import *
import random


class RandomConv2dModel:
    def __init__(self):
        self.max_layer = 10
        self.layer_generator = RandomLayers()

    def generate_layer(self, input_shape, conv_config):
        layer_num = random.randint(5, self.max_layer - 3)
        layer_count = 0
        layer_list = []
        is_embbeding = 0
        # self.layer_generator = RandomLayers()
        # layer list
        layer_map_pooling = {
            'MaxPooling2D': 28,
            'AveragePooling2D': 31,
            'GlobalMaxPooling2D': 35,
            'GlobalAveragePooling2D': 36
        }

        general_layer_map = {
            'BatchNormalization': 7
        }

        layer_map_2d = {
            'Conv2DControl': 58
        }
        general_layer_list = list(general_layer_map.keys())
        layer_2d_list = list(layer_map_2d.keys())
        layer_pooling_list = list(layer_map_pooling.keys())

        # input layer
        self.layer_generator.set_input_shape(input_shape)
        self.layer_generator.conv2d_config = conv_config

        if len(input_shape) == 3:
            self.layer_generator.now_select_layer = '2d'
            # input_layer = self.layer_generator.r_input()
            selected_layer = random.choice(layer_2d_list)
            print(selected_layer)
            if selected_layer in layer_pooling_list:
                self.layer_generator.now_select_layer = '-d'
                select_num = self.layer_generator.layer_map_pooling[selected_layer]
            elif selected_layer in layer_2d_list:
                select_num = self.layer_generator.layer_map_2d[selected_layer]

        if selected_layer in general_layer_list:
            select_num = self.layer_generator.general_layer_map[selected_layer]
        if is_embbeding == 0:
            # input_layer = self.layer_generator.layer_select(select_num)
            # model.add(input_layer)
            if select_num == 1:
                layer_list.append(4)
                layer_list.append(1)
                layer_count += 2
            else:
                layer_list.append(select_num)
                layer_count += 1
        # remain layers
        while layer_count < layer_num:
            if self.layer_generator.now_select_layer == '0d':
                selected_layer = random.choice(general_layer_list)
                print(selected_layer)

            elif self.layer_generator.now_select_layer == '2d':
                selected_layer = random.choice(general_layer_list + layer_2d_list + layer_pooling_list)
                print(selected_layer)
                if selected_layer in layer_pooling_list:
                    self.layer_generator.now_select_layer = '-d'
                    select_num = self.layer_generator.layer_map_pooling[selected_layer]
                elif selected_layer in layer_2d_list:
                    select_num = self.layer_generator.layer_map_2d[selected_layer]
            elif self.layer_generator.now_select_layer == '-d':
                selected_layer = random.choice(general_layer_list)
                print(selected_layer)

            else:
                selected_layer = random.choice(general_layer_list)
                print(selected_layer)
                if selected_layer in layer_pooling_list:
                    self.layer_generator.now_select_layer = '-d'
                    select_num = self.layer_generator.layer_map_pooling[selected_layer]

            if selected_layer in general_layer_list:
                select_num = self.layer_generator.general_layer_map[selected_layer]
            # layer = self.layer_generator.layer_select(select_num)
            if select_num == 1:
                layer_list.append(4)
                layer_list.append(1)
                layer_count += 2
            else:
                layer_list.append(select_num)
                layer_count += 1
            # model.add(layer)
        layer_list.append(4)
        layer_list.append(56)
        layer_list.append(49)
        return layer_list

    def generate_model(self, layer_list, _loss, _optimizer, layer_config=False):
        new_layer_list = []
        config_list = []
        model = Sequential()
        if layer_config:
            print("########################")
            print("########################")
            self.layer_generator.layer_config = True
        for i in range(len(layer_list)):
            try:
                if i == 0:
                    self.layer_generator.set_first_layer(1)
                    if layer_config:
                        layer = self.layer_generator.layer_select(layer_list[i], layer_config[i])
                        model.add(layer)
                    else:
                        layer, this_config = self.layer_generator.layer_select(layer_list[i])
                        model.add(layer)
                        config_list.append(this_config)
                else:
                    if layer_config:
                        layer = self.layer_generator.layer_select(layer_list[i], layer_config[i])
                        model.add(layer)
                    else:
                        layer, this_config = self.layer_generator.layer_select(layer_list[i])
                        model.add(layer)
                        config_list.append(this_config)
                new_layer_list.append(layer_list[i])
            except:
                print("skip one layer: ", layer_list[i])

        model.compile(loss=_loss,
                      optimizer=_optimizer,
                      metrics=['accuracy'])
        model.summary()
        return model, config_list, new_layer_list

    def generate_compile(self):
        _loss = r_loss()
        _optimizer = r_optimizer()
        return _loss, _optimizer


if __name__ == '__main__':
    generator = RandomConv2dModel()
    ll = generator.generate_layer((28, 28, 1), [1, 0, 0])
    _loss, _op = generator.generate_compile()
    model, config_list, new_ll = generator.generate_model(ll, _loss, _op)

