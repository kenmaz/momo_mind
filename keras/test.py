from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import mcz_input
import sys

X_train = [[[[64,128,255]]]]

model = Sequential()
model.add(Activation('softmax'))
res = model.predict(X_train)
print(res)

