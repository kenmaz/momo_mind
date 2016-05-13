#!/usr/bin/env python
#! -*- coding: utf-8 -*-

import sys
import numpy as np
import cv2
import tensorflow as tf #cv2より前にimportするとcv2.imreadになぜか失敗する(Noneを返す)
import os
import mcz_input
import mcz_model

def main(ckpt_path, filepath):
    img_files = []
    test_image = []

    if not os.path.isfile(filepath):
        print "not file"
        return

    img_files.append(filepath)
    img = cv2.imread(filepath)
    img = cv2.resize(img, (28, 28))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    test_image.append(img.flatten().astype(np.float32)/255.0)

    test_image = np.asarray(test_image)
    pixel_size = mcz_input.DST_INPUT_SIZE * mcz_input.DST_INPUT_SIZE * 3
    images_placeholder = tf.placeholder("float", shape=(None, pixel_size))
    keep_prob = tf.placeholder("float")

    logits = mcz_model.inference(images_placeholder, keep_prob, mcz_input.DST_INPUT_SIZE, mcz_input.NUM_CLASS)

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    saver.restore(sess, ckpt_path)

    for i, image in enumerate(test_image):
        print img_files[i]
        feed_dict = {images_placeholder: [image], keep_prob: 1.0 }

        softmax = logits.eval(feed_dict = feed_dict)
        print [round(n * 100.0, 1) for n in softmax[0]]

        pred = np.argmax(softmax[0])
        print mcz_input.MEMBER_NAMES[pred]

if __name__ == '__main__':
    ckpt_path = sys.argv[1]
    imgfile = sys.argv[2]
    main(ckpt_path, imgfile)

