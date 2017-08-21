#coding:utf-8

import csv
import numpy as np
from PIL import Image, ImageTk

def main():
    (X_train, Y_train) = read_data('../deeplearning/train.txt')
    (imgs_test, labels_test) = read_data('../deeplearning/test.txt')
    print(X_train.shape)
    print(Y_train.shape)

def read_data(path):
    imgs = []
    labels = []
    f = open(path, 'r')
    dataReader = csv.reader(f, delimiter=' ')
    for row in dataReader:
        path = row[0]
        img = Image.open(path, 'r')
        img = np.asarray(img)
        imgs.append(img)
        label = row[1]
        labels.append(label)
    return (np.array(imgs), np.array(labels).reshape(-1,1))

