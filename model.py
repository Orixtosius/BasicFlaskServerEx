import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def modelResults(l1, l2):
    Y = pd.read_csv('C:/Users/omur.gultekin/Desktop/R/pd_Y.csv').values
    X = pd.read_csv('C:/Users/omur.gultekin/Desktop/R/prepared_data.csv').values
    print(f'\nShape of data X : {X.shape} Y : {Y.shape}\n')

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.25)

    print(f'\nShape of data X : Test {X_test.shape} Train {X_train.shape}')
    print(f'Shape of data Y : Test {Y_test.shape} Train {Y_train.shape}\n')

    i = Input(shape=X_train.shape[1])
    x = Dense(l1, activation = 'relu')(i)
    x = Dense(l2, activation = 'relu')(x)
    x = Dense(1, activation = 'relu')(x)

    model = Model(i,x)
    model.compile(loss = 'mse', optimizer = 'adam')

    r = model.fit(X_train, Y_train, epochs = 100, validation_data = (X_test, Y_test))
    return r
