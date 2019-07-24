from random_layer import *
import random
from keras.models import Sequential


class RandomRNNModel:
    def __init__(self):
        self.max_layer = 10
        self.layer_generator = RandomLayers()

    def generate_layer(self, input_shape):
        layer_num = random.randint(3, self.max_layer - 3)
        layer_count = 0
        layer_list = []
        layer_rnn_list = list(self.layer_generator.layer_map_rnn.keys())
        general_layer_list = list(self.layer_generator.general_layer_map.keys())
        layer_3d_list = list(self.layer_generator.layer_map_3d.keys())
        self.layer_generator.set_input_shape(input_shape)
        if len(input_shape) == 2:
            selected_layer = random.choice(layer_rnn_list)
            self.layer_generator.now_select_layer = '-d'
            print(selected_layer)
            select_num = self.layer_generator.layer_map_rnn[selected_layer]
            layer_list.append(select_num)
        elif len(input_shape) == 4:
            layer_list.append(44)
            self.layer_generator.now_select_layer = '3d'
        layer_count += 1
        while layer_count < layer_num:
            if self.layer_generator.now_select_layer == '-d':
                selected_layer = random.choice(general_layer_list)
                select_num = self.layer_generator.general_layer_map[selected_layer]

            elif self.layer_generator.now_select_layer == '3d':
                selected_layer = random.choice(layer_3d_list + general_layer_list)
                if selected_layer in layer_3d_list:
                    select_num = self.layer_generator.layer_map_3d[selected_layer]
                else:
                    select_num = self.layer_generator.general_layer_map[selected_layer]
            print(selected_layer)
            if select_num == 1:
                layer_list.append(4)
                layer_list.append(1)
                layer_count += 2
            else:
                layer_list.append(select_num)
                layer_count += 1
        layer_list.append(4)
        layer_list.append(56)
        layer_list.append(49)
        return layer_list

    def generate_model(self, layer_list, _loss, _optimizer, layer_config=False):
        new_layer_list = []
        config_list = []
        model = Sequential()
        if layer_config:
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
    generator = RandomRNNModel()
    ll = generator.generate_layer((None, 28, 28, 1))
    _loss, _op = generator.generate_compile()
    model, config_list, new_ll = generator.generate_model(ll, _loss, _op)

