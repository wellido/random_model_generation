from random_layer import RandomLayers
from keras.models import Sequential
from utils import *
import random


class RandomModel:
    def __init__(self):
        self.max_layer = 20
        self.layer_generator = RandomLayers()

    def generate_layer(self, input_shape):
        layer_num = random.randint(10, self.max_layer - 3)
        layer_count = 0
        layer_list = []
        is_embbeding = 0
        # self.layer_generator = RandomLayers()
        # layer list
        general_layer_list = list(self.layer_generator.general_layer_map.keys())
        layer_0d_list = list(self.layer_generator.layer_map_0d.keys())
        layer_1d_list = list(self.layer_generator.layer_map_1d.keys())
        layer_2d_list = list(self.layer_generator.layer_map_2d.keys())
        layer_3d_list = list(self.layer_generator.layer_map_3d.keys())
        layer_rnn_list = list(self.layer_generator.layer_map_rnn.keys())
        layer_pooling_list = list(self.layer_generator.layer_map_pooling.keys())
        list_upsample_1d = ['UpSampling1D']
        list_upsample_2d = ['UpSampling2D']
        list_upsample_3d = ['UpSampling3D']

        # input layer
        self.layer_generator.set_input_shape(input_shape)
        self.layer_generator.set_first_layer(1)
        if len(input_shape) == 1:
            self.layer_generator.now_select_layer = '0d'
            layer_list.append(54)
            is_embbeding = 1

        elif len(input_shape) == 2:
            self.layer_generator.now_select_layer = '1d'
            selected_layer = random.choice(list(set(layer_0d_list + layer_1d_list + general_layer_list) -
                                                set(list_upsample_1d)))
            print(selected_layer)
            if selected_layer in layer_0d_list:
                self.layer_generator.now_select_layer = '-d'
                select_num = self.layer_generator.layer_map_0d[selected_layer]
            elif selected_layer in layer_pooling_list:
                self.layer_generator.now_select_layer = '-d'
                select_num = self.layer_generator.layer_map_pooling[selected_layer]
            elif selected_layer in layer_1d_list:
                select_num = self.layer_generator.layer_map_1d[selected_layer]
        elif len(input_shape) == 3:
            self.layer_generator.now_select_layer = '2d'
            # input_layer = self.layer_generator.r_input()
            selected_layer = random.choice(list(set(layer_0d_list + layer_2d_list + general_layer_list)
                                                - set(list_upsample_2d)))
            print(selected_layer)
            if selected_layer in layer_0d_list:
                self.layer_generator.now_select_layer = '-d'
                select_num = self.layer_generator.layer_map_0d[selected_layer]
            elif selected_layer in layer_pooling_list:
                self.layer_generator.now_select_layer = '-d'
                select_num = self.layer_generator.layer_map_pooling[selected_layer]
            elif selected_layer in layer_2d_list:
                select_num = self.layer_generator.layer_map_2d[selected_layer]
        else:
            self.layer_generator.now_select_layer = '3d'
            selected_layer = random.choice(list(set(layer_0d_list + layer_3d_list + general_layer_list) -
                                                set(list_upsample_3d)))
            print(selected_layer)
            if selected_layer in layer_0d_list:
                self.layer_generator.now_select_layer = '-d'
                select_num = self.layer_generator.layer_map_0d[selected_layer]
            elif selected_layer in layer_pooling_list:
                self.layer_generator.now_select_layer = '-d'
                select_num = self.layer_generator.layer_map_pooling[selected_layer]
            elif selected_layer in layer_3d_list:
                select_num = self.layer_generator.layer_map_3d[selected_layer]
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
        self.layer_generator.set_first_layer(0)
        # remain layers
        while layer_count < layer_num:
            if self.layer_generator.now_select_layer == '0d':
                if layer_count == layer_num - 1:
                    selected_layer = random.choice(general_layer_list)
                else:
                    selected_layer = random.choice(general_layer_list + layer_rnn_list)
                print(selected_layer)
                if selected_layer in layer_rnn_list:
                    self.layer_generator.now_select_layer = '-d'
                    select_num = self.layer_generator.layer_map_rnn[selected_layer]
                    # layer = self.layer_generator.layer_select(select_num)
                    # model.add(layer)
                    # layer_list.append(layer)
            elif self.layer_generator.now_select_layer == '1d':
                selected_layer = random.choice(layer_0d_list + layer_1d_list + general_layer_list)
                print(selected_layer)
                if selected_layer in layer_0d_list:
                    self.layer_generator.now_select_layer = '-d'
                    select_num = self.layer_generator.layer_map_0d[selected_layer]
                if selected_layer in layer_pooling_list:
                    self.layer_generator.now_select_layer = '-d'
                    select_num = self.layer_generator.layer_map_pooling[selected_layer]
                if selected_layer in layer_1d_list:
                    select_num = self.layer_generator.layer_map_1d[selected_layer]
            elif self.layer_generator.now_select_layer == '2d':
                selected_layer = random.choice(layer_0d_list + layer_2d_list + general_layer_list)
                print(selected_layer)
                if selected_layer in layer_0d_list:
                    self.layer_generator.now_select_layer = '-d'
                    select_num = self.layer_generator.layer_map_0d[selected_layer]
                if selected_layer in layer_pooling_list:
                    self.layer_generator.now_select_layer = '-d'
                    select_num = self.layer_generator.layer_map_pooling[selected_layer]
                if selected_layer in layer_2d_list:
                    select_num = self.layer_generator.layer_map_2d[selected_layer]
            elif self.layer_generator.now_select_layer == '-d':
                selected_layer = random.choice(general_layer_list)
                print(selected_layer)

            else:
                selected_layer = random.choice(layer_0d_list + layer_3d_list + general_layer_list)
                print(selected_layer)
                if selected_layer in layer_0d_list:
                    self.layer_generator.now_select_layer = '-d'
                    select_num = self.layer_generator.layer_map_0d[selected_layer]
                if selected_layer in layer_pooling_list:
                    self.layer_generator.now_select_layer = '-d'
                    select_num = self.layer_generator.layer_map_pooling[selected_layer]
                if selected_layer in layer_3d_list:
                    select_num = self.layer_generator.layer_map_3d[selected_layer]
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
    test = RandomModel()
    ll = test.generate_layer((28, 28, 1))
    model = test.generate_model(ll)

