
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras import losses

def create_model(number_class = 42, input_size = 9562):
    number_class = number_class
    model = Sequential([
          layers.InputLayer(input_shape = (input_size, )),
          layers.Dense(100, activation = 'relu'),
          layers.Dense(number_class, activation = 'softmax')
    ])
    model.compile(optimizer=optimizers.Adam(0.001), loss=losses.SparseCategoricalCrossentropy(), metrics='accuracy')
    return model