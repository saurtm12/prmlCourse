import numpy as np
from readData import readData
from model import modelDef


def train():
    readData()
    X_train = np.load("X_train.npy")
    X_train = X_train.reshape(X_train.shape[0], 862, 40, 1)

    y_train = np.load("y_train.npy").astype("float64")
    y_train = to_categorical(y_train)

    X_test = np.load("X_test_resample.npy")
    X_test = X_test.reshape(X_test.shape[0], 862, 40, 1)

    model = modelDef()
    model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2, verbose=1)
    return model.predict(X_test, verbose=1)

def predict():
    prediction = train()
    prediction = prediction.reshape(4512)
    with open("submission_kaggle.csv", "w") as fp: 
        fp.write("Id,Predicted\n") 
        for idx in range(4512): 
            fp.write(f"{idx},{prediction[idx]}\n")


predict()
