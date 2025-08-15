from keras.models import Sequential
from keras.layers import Dense, Flatten, Input, Conv2D, MaxPooling2D, BatchNormalization


class Configuracion1:
    def __init__(self):
        self.name = "Configuracion1"
        self.N = 10
        self.K = 5
        self.P = 0
        self.S = 2


class Configuracion2:
    def __init__(self):
        self.name = "Configuracion2"
        self.N = 10
        self.K = 5
        self.P = 0
        self.S = 1


def get_model(config):
    model = Sequential()
    model.add(
        Conv2D(
            config.N,
            config.K,
            strides=config.S,
            padding="valid",
            activation="relu",
            input_shape=(50, 50, 1),
        )
    )
    if config.name == "Configuracion2":
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Flatten())
    model.add(Dense(10, activation="softmax"))

    return model


print("Configuracion 1")
get_model(Configuracion1()).summary()
print("Configuracion 2")
get_model(Configuracion2()).summary()
