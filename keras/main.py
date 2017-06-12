#coding:utf-8

import csv
import numpy as np
import cv2

def main():
    (imgs_train, labels_train) = read_data('/Users/kentaro.matsumae/Projects/momo_mind/deeplearning/train.txt')
    (imgs_test, labels_test) = read_data('/Users/kentaro.matsumae/Projects/momo_mind/deeplearning/test.txt')
    print(imgs_train[0].shape) #(3,112,112)
    print(labels_train[0]) #()

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
