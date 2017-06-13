#coding:utf-8

import csv
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

def main():
    (X_train, Y_train) = read_data('/Users/kentaro.matsumae/Projects/momo_mind/deeplearning/train.txt')
    (imgs_test, labels_test) = read_data('/Users/kentaro.matsumae/Projects/momo_mind/deeplearning/test.txt')

    model = Sequential()
    # input: 112x112 images with 3 channels -> (3, 112, 112) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(3, 112, 112)))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    # Note: Keras does automatic shape inference.
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    model.fit(X_train, Y_train, batch_size=32, nb_epoch=1)

def read_data(path):
    imgs = []
    labels = []
    f = open(path, 'r')
    dataReader = csv.reader(f, delimiter=' ')
    for row in dataReader:
        path = row[0]
        img = cv2.imread(path)
        imgs.append(img)
        label = row[1]
        labels.append(label)
    return (np.array(imgs), np.array(labels))

main()
