from random_layer import RandomLayers
from keras.models import Sequential

from utils import *
import random


class Random1layerModel:
    def __init__(self):
        self.max_layer = 10
        self.layer_generator = RandomLayers()

    def generate_layer(self, input_shape, input_layer_num):

        layer_list = []
        # input layer
        self.layer_generator.set_input_shape(input_shape)

        select_num = input_layer_num
        layer_list.append(select_num)
        if select_num == 4:
            layer_list.append(56)
        else:
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
    generator = Random1layerModel()
    ll = generator.generate_layer((28, 28, 1))
    _loss, _op = generator.generate_compile()
    model, config_list, new_ll = generator.generate_model(ll, _loss, _op)


