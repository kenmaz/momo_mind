from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model
import mcz_input
import sys
import numpy as np

(X_test, y_test)= mcz_input.read_data('../deeplearning/predict.txt')

X_test = X_test.astype('float32')
X_test /= 255

nb_classes = 5
Y_test = np_utils.to_categorical(y_test, nb_classes)
print(Y_test)

model = load_model('model.h5')
res = model.predict(X_test)
print([np.argmax(i) for i in res])



