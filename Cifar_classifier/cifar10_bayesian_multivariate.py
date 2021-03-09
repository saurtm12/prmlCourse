import pickle
import numpy as np
import matplotlib.pyplot as plt
from random import random
from skimage.transform import rescale, resize, downscale_local_mean
from scipy.stats import norm
import scipy
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

dataY = np.array(dataY)
Y = np.array(Y)

labeldict = unpickle('./cifar-10-batches-py/batches.meta')
label_names = labeldict["label_names"]
X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint")
X_mean = np.zeros((10000,3))
NUM_OF_CLASS = 10
size = 1 
def cifar10_color(X):
    sz= 1
    totalValue = sz * sz * 3
    X_mean = np.zeros((X.shape[0], totalValue))
    for i in range(X.shape[0]):
        img = X[i]
        img_nxn = resize(img, (sz, sz), preserve_range=True)
        r_vals = img_nxn[:, :, 0].reshape(sz * sz)
        g_vals = img_nxn[:, :, 1].reshape(sz * sz)
        b_vals = img_nxn[:, :, 2].reshape(sz * sz)
        X_mean[i, :] = np.concatenate((r_vals, g_vals, b_vals))
    return X_mean

def grouping_images_to_class(Xf, Y, sz):
    imgClasses = {}
    for i in range(10):
        imgClasses[i] = np.empty((0, sz  * sz * 3))
    for i in range(Xf.shape[0]):
        imgClasses[Y[i]] = np.concatenate((imgClasses[Y[i]], [Xf[i]]))
    return imgClasses
    
def cifar10_bayes_learn(Xf, Y):
    sz = 1
    imgClasses = grouping_images_to_class(Xf, Y, sz)
    mu = [np.mean(imgClasses[i], axis=0) for i in range(10)]
    cov = [np.cov(imgClasses[i].T) for i in range(10)]
    prior = [len(imgClasses[i])/len(Y) for i in range(10)]
    return np.array(mu), np.array(cov), np.array(prior)

def cifar10_classifier_bayes(x,mu,sigma,p):
    prob = [0 for i in range(10)]
    for i in range(NUM_OF_CLASS):
        prob[i] = scipy.stats.multivariate_normal.pdf(x,mu[i],sigma[i])
    return np.argmax([prob[i]*p[i] for i in range(NUM_OF_CLASS)])

X = cifar10_color(X)
dataX = cifar10_color(dataX)
mu, sigma, prior = cifar10_bayes_learn(dataX,dataY)
pred = [cifar10_classifier_bayes(x,mu,sigma,prior) for x in X]

def class_acc(pred,gt):
    correct = 0
    for i in range(len(pred)):
        if pred[i] == gt[i]:
            correct +=1
    return correct/len(pred)

#expect 0.2474
print(class_acc(pred,Y))
#This result is better because the color actually relates to each other