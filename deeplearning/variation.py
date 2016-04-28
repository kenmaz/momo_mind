#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import cv2

IMAGE_SIZE = 28

def show(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def zoom(src, ratio):
    src_w = src.shape[0]
    src_h = src.shape[1]
    z_w, z_h = (int(src_w * ratio), int(src_h * ratio))
    img_z = cv2.resize(src, (z_w, z_h))
    d_w, d_h = (int((z_w - src_w)/2), int((z_h - src_h)/2))
    img_zc = img_z[d_w:d_w+src_w, d_h:d_h + src_h]
    return img_zc

def variation(src):
    imgs = []
    zoom_ratios = [1.0, 1.1, 1.2]
    flips = [True, False]
    for flip in flips:
        for ratio in zoom_ratios:
            print 'zoom:%f flip:%s' % (ratio, flip)
            img = zoom(src, ratio)
            if flip:
                img = cv2.flip(img, 1)
            imgs.append(img)
    return imgs

if __name__ == '__main__':
    filepath = sys.argv[1]
    print filepath
    src = cv2.imread(filepath)
    imgs = variation(src)
    for img in imgs:
        show(img)

