import pickle
import numpy as np
import matplotlib.pyplot as plt
from random import random
from skimage.transform import rescale, resize, downscale_local_mean
from scipy.stats import norm

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
X_mean = np.zeros((10000,3))
NUM_OF_CLASS = 10

def cifar10_color(X):
    sz= 1
    totalValue = sz * sz * 3
    X_mean = np.zeros((X.shape[0], totalValue))
    for i in range(X.shape[0]):
        img = X[i]
        img_1x1 = resize(img, (sz, sz), preserve_range=True)
        r_vals = img_1x1[:, :, 0].reshape(sz * sz)
        g_vals = img_1x1[:, :, 1].reshape(sz * sz)
        b_vals = img_1x1[:, :, 2].reshape(sz * sz)
        X_mean[i, :] = np.concatenate((r_vals, g_vals, b_vals))
    return X_mean

def cifar_10_naivebayes_learn(Xf,Y):
    mu = [[0,0,0] for i in range(NUM_OF_CLASS)]
    var = np.array(mu, copy =True)
    var.astype("uint8")
    prior = [0 for i in range(NUM_OF_CLASS)]

    for i in range(len(Y)):
        idx = Y[i]
        prior[idx] += 1
        mu[idx] = np.add(mu[idx],Xf[i])
    for i in range(NUM_OF_CLASS):
        mu[i] = mu[i]/prior[i]
      
    for i in range(len(Y)):
        for j in range(3):
            var[Y[i]][j] += (Xf[i][j]-mu[Y[i]][j])**2
    var = [var[i]/prior[i] for i in range(NUM_OF_CLASS)]
    sigma = [[x**(1/2) for x in typ] for typ in var]    
    prior = [i/len(Y) for i in prior]

    return mu, sigma, prior
def cifar10_classifier_bayes(x,mu,sigma,p):
    prob = np.array(mu,copy=True)
    for i in range(NUM_OF_CLASS):
        for j in range(3):
            prob[i][j] = norm.pdf(x[j], loc=mu[i][j], scale=sigma[i][j])
    probBelong = [np.prod(prob[i])*p[i] for i in range(NUM_OF_CLASS)]
    return np.argmax([i for i in probBelong])

X = cifar10_color(X)
dataX = cifar10_color(dataX)
mu, sigma, prior = cifar_10_naivebayes_learn(dataX,dataY)
pred = [cifar10_classifier_bayes(x,mu,sigma,prior) for x in X]
def class_acc(pred,gt):
    correct = 0
    for i in range(len(pred)):
        if pred[i] == gt[i]:
            correct +=1
    return correct/len(pred)

#expect 0.1954
print(class_acc(pred,Y))
