# random_model_generation

Usage:
```python
from random_model import RandomModel
generator = RandomModel()
layer_list = generator.generate_layer((28, 28, 1))
model = generator.generate_model(layer_list)
```
