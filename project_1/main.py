import cv2
import imageio
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD

def shuffle(trdata, trlabel):
    p = np.random.permutation(len(trdata))
    return trdata[p], trlabel[p]


def readData():
    trdataX = []
    trdataY = []
    for i in range(10):
        dir = f'./train/train/{i+1}/'
        for j in range(1,6001):
            img = imageio.imread(f"{dir}{j:05}.jpg")
            trdataX.append(img)
            trdataY.append(i)

    testX = []
    dir = f'./test/test/'
    for j in range(10000):
        img = imageio.imread(f"{dir}{j:05}.jpg")
        testX.append(img)

    trdataX = np.array(trdataX)
    testX = np.array(testX)
    trdataY = np.array(trdataY)

    trdataX = trdataX.reshape(60000, 28, 28, 1).astype("float32") / 255.0
    testX = testX.reshape(10000, 28, 28, 1).astype("float32") / 255.0
    trdataY = to_categorical(trdataY)

    #shuffle data before train
    trdataX, trdataY = shuffle(trdataX, trdataY)

    return trdataX, trdataY, testX


def modelDef():
    '''
    This model has architecture of VGG16 convolutional neural network for classifying 10 letters(categories)
    '''
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same"))
    model.add(Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same"))
    model.add(Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same"))
    model.add(Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(100, activation="relu", kernel_initializer="he_uniform"))
    model.add(Dense(10, activation='softmax'))

    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def train():
    X, Y, testX = readData()
    model = modelDef()
    model.fit(X, Y, epochs=120, batch_size=64, validation_split=0.2, verbose=1)
    return model.predict_classes(testX, verbose=1)


def predict():
    prediction = train()
    with open("submission_kaggle.csv", "w") as fp: 
        fp.write("Id,Category\n") 
        for idx in range(10000): 
            fp.write(f"{idx:05},{prediction[idx]+1}\n")

predict()
