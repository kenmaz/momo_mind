#!/usr/bin/env python
#! -*- coding: utf-8 -*-

import sys
import numpy as np
import cv2
import tensorflow as tf #cv2より前にimportするとcv2.imreadになぜか失敗する(Noneを返す)
import os
import mcz_input
import mcz_model

def main(filepaths, ckpt_path = None ):
    if len(filepaths) == 0:
        print "empty"
        return []

    img_files = []
    test_image = []

    for filepath in filepaths:
        if not os.path.isfile(filepath):
            print "not file:", filepath
            continue
        img_files.append(filepath)
        img = cv2.imread(filepath)
        img = cv2.resize(img, (28, 28))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        test_image.append(img.flatten().astype(np.float32)/255.0)

    tf.reset_default_graph()

    test_image = np.asarray(test_image)
    pixel_size = mcz_input.DST_INPUT_SIZE * mcz_input.DST_INPUT_SIZE * 3
    images_placeholder = tf.placeholder("float", shape=(None, pixel_size))
    keep_prob = tf.placeholder("float")

    logits = mcz_model.inference(images_placeholder, keep_prob, mcz_input.DST_INPUT_SIZE, mcz_input.NUM_CLASS)

    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())

    if ckpt_path:
        saver = tf.train.Saver()
        saver.restore(sess, ckpt_path)

    res = []

    for i, image in enumerate(test_image):
        print img_files[i]

        feed_dict = {images_placeholder: [image], keep_prob: 1.0 }
        softmax = logits.eval(feed_dict = feed_dict)

        result = softmax[0]
        rates = [round(n * 100.0, 1) for n in result]
        print rates

        pred = np.argmax(result)
        print mcz_input.MEMBER_NAMES[pred]

        members = []
        for idx, rate in enumerate(rates):
            name = mcz_input.MEMBER_NAMES[idx]
            members.append({
                'name': name,
                'rate': rate
            })
        rank = sorted(members, key=lambda x: x['rate'], reverse=True)

        res.append({
            'file': img_files[i],
            'rank': rank
        })

    sess.close()
    return res

if __name__ == '__main__':
    ckpt_path = sys.argv[1]
    imgfile = sys.argv[2]
    main([imgfile], ckpt_path)
