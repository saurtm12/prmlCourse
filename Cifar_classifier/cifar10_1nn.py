import pickle
import numpy as np
import matplotlib.pyplot as plt
from random import random
import random
import math
from scipy.spatial import distance

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

datadict = unpickle('./cifar-10-batches-py/data_batch_1')
dataX = datadict["data"]
dataY = datadict["labels"]
dataX = dataX.astype("uint8")
dataY = np.array(dataY)

datadict = unpickle('./cifar-10-batches-py/test_batch')
X = datadict["data"]
Y = datadict["labels"]
X = X.astype("uint8")
Y = np.array(Y)

labeldict = unpickle('./cifar-10-batches-py/batches.meta')
label_names = labeldict["label_names"]

# def get_to_know():
#     X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
#     Y = np.array(Y)
#     for i in range(X.shape[0]):
#     # Show some images randomly
#         if random() > 0.999:
#             plt.figure(1)
#             plt.clf()
#             plt.imshow(X[i])
#             plt.title(f"Image {i} label={label_names[Y[i]]} (num {Y[i]})")
#             plt.pause(1)


def class_acc(pred,gt):
    correct = 0
    for i in range(len(pred)):
        if pred[i] == gt[i]:
            correct +=1
    return correct/len(pred)

def cifar10_classifier_random(x):
    result =[]
    for i in range(len(x)):
        result.append(random.randint(0,9))
    return resultfl

def cifar10_classifier_1nn(x, trdata, trlabels):
    idx = np.argmin([distance.correlation(x,img) for img in trdata])
    return trlabels[idx]

pred = [cifar10_classifier_1nn(x_test, dataX,dataY) for x_test in X]
acc = class_acc(pred,Y)
print('accuracy: ',acc)



        