#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import tensorflow as tf
import momo_input
import main3

def main(ckpt_path):
    with tf.Graph().as_default():
        images, labels, filename = momo_input.load_data_for_test(['test.txt'], 88)
        print 'start', images, labels
        keep_prob = tf.placeholder("float")

        logits = main3.inference(images, keep_prob)
        acc = main3.accuracy(logits, labels)

        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, ckpt_path)
        tf.train.start_queue_runners(sess)

        acc_res, filename_res = sess.run([acc, filename], feed_dict={keep_prob: 0.5})
        print acc_res

if __name__ == '__main__':
    ckpt_path = sys.argv[1]
    print ckpt_path
    main(ckpt_path)

