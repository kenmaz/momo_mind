#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import tensorflow as tf
import momo_input
import main3
import numpy as np

def main(ckpt_path):
    with tf.Graph().as_default():
        images, labels, filename = momo_input.load_data_for_test(['test.txt'], 100)
        print 'start', images, labels
        keep_prob = tf.placeholder("float")

        logits = main3.inference(images, keep_prob)
        acc = main3.accuracy(logits, labels)

        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, ckpt_path)
        tf.train.start_queue_runners(sess)

        goods = []
        bads = []
        acc_res, filename_res, actual_res, expect_res = sess.run([acc, filename, logits, labels], feed_dict={keep_prob: 0.5})
        for idx, (act, exp) in enumerate(zip(actual_res, expect_res)):
            if np.argmax(act) == np.argmax(exp):
                goods.append(filename_res[idx])
            else:
                bads.append(filename_res[idx])
        print 'good'
        for f in goods:
            print 'cp',f,'out_goods'
        print 'bad'
        for f in bads:
            print 'cp',f,'out_bads'
        #correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))

        print 'accuracy', acc_res

if __name__ == '__main__':
    ckpt_path = sys.argv[1]
    print ckpt_path
    main(ckpt_path)

