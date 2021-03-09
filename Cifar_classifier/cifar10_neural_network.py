import pickle
import numpy as np
import matplotlib.pyplot as plt
from random import random
from skimage.transform import rescale, resize, downscale_local_mean
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical


def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

def readTrainData():
    firstBatch = unpickle('cifar-10-batches-py/data_batch_1')
    trdata = firstBatch["data"]
    trlabels = firstBatch["labels"]

    for i in range(2, 6):
        path = f'cifar-10-batches-py/data_batch_{i}'
        data = unpickle(path)
        trdata = np.append(trdata, data["data"], axis=0)
        trlabels = np.append(trlabels, data["labels"], axis=0)
    return trdata, trlabels
dataX,dataY = readTrainData()
dataX = dataX.reshape(50000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")

datadict = unpickle('./cifar-10-batches-py/test_batch')
X = datadict["data"]
Y = datadict["labels"]

labeldict = unpickle('./cifar-10-batches-py/batches.meta')
label_names = labeldict["label_names"]
Y = np.array(Y)
X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint")
NUM_OF_CLASS = 10

def normalize(trdata, X):
    trdata_norm = trdata.astype('float32')
    X_norm = X.astype('float32')
    trdata_norm = trdata_norm/255.0
    X_norm = X_norm/255.0
    return trdata_norm, X_norm
def modelDef():
    model = Sequential()

    model.add(Conv2D(32,(3,3), activation='relu',kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(32,(3,3), activation='relu',kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64,(3,3), activation='relu',kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(64,(3,3), activation='relu',kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128,(3,3), activation='relu',kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128,(3,3), activation='relu',kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.2))
    
    model.add(Flatten())

    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.2))
    model.add(Dense(10,activation='sigmoid'))
    opt = tf.keras.optimizers.SGD(lr=0.001, momentum=0.9)

    model.compile(optimizer=opt,loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def class_acc(pred,gt):
    correct = 0
    for i in range(len(pred)):
        if pred[i] == gt[i]:
            correct +=1
    return correct/len(pred)
    
def train():
    model = modelDef()
    trdata,X_test = normalize(dataX,X)
    model.fit(trdata,dataY,epochs=100,batch_size = 64, validation_split=0.2, verbose=1)
    _,acc = model.evaluate(X_test,Y, verbose=1)
    #epoch = 100 then validate_accuracy: 73.66 -> accuracy in test : 72.90 
    print("acc : ",acc)

Y = to_categorical(Y)
dataY = to_categorical(dataY)
train()