#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import time
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.python.platform
from datetime import datetime
import random

import mcz_input
import mcz_model

NUM_CLASSES = 5
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE * 3

LOGDIR = '/tmp/data.%s' % datetime.now().isoformat()
print LOGDIR

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('train', 'train_bk2.txt', 'File name of train data')
#flags.DEFINE_string('train', 'train_1.txt', 'File name of train data')
flags.DEFINE_string('test', 'test_bk2.txt', 'File name of train data')
#flags.DEFINE_string('test', 'test_osaretai.txt', 'File name of train data')
flags.DEFINE_string('train_dir', LOGDIR, 'Directory to put the training data.')
#flags.DEFINE_integer('max_steps', 200, 'Number of steps to run trainer.')
flags.DEFINE_integer('max_steps', 1000, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 120, 'Batch size Must divide evenly into the dataset sizes.')
flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')
#flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')

def main():
    f = open(FLAGS.train, 'r')
    train_image = []
    train_label = []
    tuple_list = []
    for line in f:
        line = line.rstrip()
        l = line.split()
        #print l;
        src = cv2.imread(l[0])
        imgs = mcz_input.variation(src)
        #imgs = [src]
        for img in imgs:
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            img_flt = img.flatten().astype(np.float32)/255.0
            tmp = np.zeros(NUM_CLASSES)
            tmp[int(l[1])] = 1
            tuple_list.append((img_flt, tmp))
    random.shuffle(tuple_list)
    for (img, label) in tuple_list:
        train_image.append(img)
        train_label.append(label)
        #print 'train: %s' % label

    train_image = np.asarray(train_image)
    train_label = np.asarray(train_label)
    f.close()

    f = open(FLAGS.test, 'r')
    test_image = []
    test_label = []
    for line in f:
        line = line.rstrip()
        l = line.split()
        img = cv2.imread(l[0])
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        test_image.append(img.flatten().astype(np.float32)/255.0)
        tmp = np.zeros(NUM_CLASSES)
        tmp[int(l[1])] = 1
        test_label.append(tmp)
    test_image = np.asarray(test_image)
    test_label = np.asarray(test_label)
    f.close()

    with tf.Graph().as_default():
        images_placeholder = tf.placeholder("float", shape=(None, IMAGE_PIXELS))
        labels_placeholder = tf.placeholder("float", shape=(None, NUM_CLASSES))
        keep_prob = tf.placeholder("float")

        logits = mcz_model.inference(images_placeholder, keep_prob)
        loss_value = mcz_model.loss(logits, labels_placeholder)
        train_op = mcz_model.training(loss_value, FLAGS.learning_rate)
        acc = mcz_model.accuracy(logits, labels_placeholder)

        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph_def)

        for step in range(FLAGS.max_steps):
            for i in range(len(train_image)/FLAGS.batch_size):
                batch = FLAGS.batch_size*i
                start_time = time.time()
                _, loss_result = sess.run([train_op, loss_value], feed_dict={
                  images_placeholder: train_image[batch:batch+FLAGS.batch_size],
                  labels_placeholder: train_label[batch:batch+FLAGS.batch_size],
                  keep_prob: 0.5})
                duration = time.time() - start_time

                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
                print (format_str % (datetime.now(), step, loss_result, examples_per_sec, sec_per_batch))

            train_accuracy = sess.run(acc, feed_dict={
                images_placeholder: train_image,
                labels_placeholder: train_label,
                keep_prob: 1.0})
            print "step %d, training accuracy %g"%(step, train_accuracy)

            summary_str = sess.run(summary_op, feed_dict={
                images_placeholder: train_image,
                labels_placeholder: train_label,
                keep_prob: 1.0})
            summary_writer.add_summary(summary_str, step)

            print "test accuracy %g"%sess.run(acc, feed_dict={
                images_placeholder: test_image,
                labels_placeholder: test_label,
                keep_prob: 1.0})

            save_path = saver.save(sess, "model.ckpt")

if __name__ == '__main__':
    main()

