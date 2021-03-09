import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Reshape
from tensorflow.keras.optimizers import SGD, Adam

def modelDef():
    '''
    Recurrent convolutional neural network for detecting if there exists bird sounds in the audio. Output of the last layer is the probability if the bird sounds is in the audio.
    The model contains 3 convolutional layers to extract the features, then a recurrent network and finally a fully connected
    '''
    model = Sequential()

    model.add(Conv2D(96, (5, 5), activation="relu", kernel_initializer="he_uniform", padding="same", input_shape=(862, 40, 1)))
    model.add(MaxPooling2D((5, 5)))
    model.add(Dropout(0.25))

    model.add(Conv2D(96, (5, 5), activation="relu", kernel_initializer="he_uniform", padding="same"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(96, (5, 5), activation="relu", kernel_initializer="he_uniform", padding="same"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(96, (5, 5), activation="relu", kernel_initializer="he_uniform", padding="same"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Reshape((96, -1)))

    model.add(GRU(96))
    model.add(Dropout(0.25))

    model.add(GRU(96))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation="relu", kernel_initializer="he_uniform"))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation="sigmoid"))

    opt = Adam(learning_rate=0.01)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    return model
