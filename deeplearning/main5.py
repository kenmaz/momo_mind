#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.python.platform
from datetime import datetime
import time
import random

import variation
import momo_input2 as momo_input

NUM_CLASSES = 5
IMAGE_SIZE = momo_input.DST_INPUT_SIZE
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE * 3

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

LOGDIR = '/tmp/data.%s' % datetime.now().isoformat()
print LOGDIR

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('train', 'train.txt', 'File name of train data')
flags.DEFINE_string('test', 'test.txt', 'File name of train data')
flags.DEFINE_string('train_dir', LOGDIR, 'Directory to put the training data.')
flags.DEFINE_integer('max_steps', 1000000, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 120, 'Batch size Must divide evenly into the dataset sizes.')
flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')
#flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')

BATCH_SIZE = FLAGS.batch_size

def inference(images):
    def _variable_with_weight_decay(name, shape, stddev, wd):
        var = tf.get_variable(name, shape=shape, initializer=tf.truncated_normal_initializer(stddev=stddev))
        if wd:
            weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    def _activation_summary(x):
        tensor_name = x.op.name
        tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

    with tf.variable_scope('conv1') as scope:
        kernel = tf.get_variable('weights', shape=[3, 3, 3, 32], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', shape=[32], initializer=tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv1)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    with tf.variable_scope('conv2') as scope:
        kernel = tf.get_variable('weights', shape=[3, 3, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', shape=[64], initializer=tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv2)
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    with tf.variable_scope('conv3') as scope:
        kernel = tf.get_variable('weights', shape=[3, 3, 64, 128], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', shape=[128], initializer=tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv3)
        pool3 = tf.nn.max_pool(conv3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

    with tf.variable_scope('conv4') as scope:
        kernel = tf.get_variable('weights', shape=[3, 3, 128, 256], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', shape=[256], initializer=tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv4)
        pool4 = tf.nn.max_pool(conv4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

    with tf.variable_scope('fc5') as scope:
        dim = 1
        for d in pool4.get_shape()[1:].as_list():
            dim *= d
        reshape = tf.reshape(pool4, [BATCH_SIZE, dim])
        weights = _variable_with_weight_decay('weights', shape=[dim, 1024], stddev=0.02, wd=0.005)
        biases = tf.get_variable('biases', shape=[1024], initializer=tf.constant_initializer(0.0))
        fc5 = tf.nn.relu(tf.nn.bias_add(tf.matmul(reshape, weights), biases), name=scope.name)
        _activation_summary(fc5)

    with tf.variable_scope('fc6') as scope:
        weights = _variable_with_weight_decay('weights', shape=[1024, 256], stddev=0.02, wd=0.005)
        biases = tf.get_variable('biases', shape=[256], initializer=tf.constant_initializer(0.0))
        fc6 = tf.nn.relu(tf.nn.bias_add(tf.matmul(fc5, weights), biases), name=scope.name)
        _activation_summary(fc6)

    with tf.variable_scope('fc7') as scope:
        weights = tf.get_variable('weights', shape=[256, NUM_CLASSES], initializer=tf.truncated_normal_initializer(stddev=0.02))
        biases = tf.get_variable('biases', shape=[NUM_CLASSES], initializer=tf.constant_initializer(0.0))
        fc7 = tf.nn.bias_add(tf.matmul(fc6, weights), biases, name=scope.name)
        _activation_summary(fc7)

    return fc7

def inference_(x_image, keep_prob):

    # 重みを標準偏差0.1の正規分布で初期化
    def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

    # バイアスを標準偏差0.1の正規分布で初期化
    def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

    # 畳み込み層の作成
    def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    # プーリング層の作成
    def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')

    def tf_print(tensor, name):
        print name, tensor
        return tensor
        #return tf.Print(tensor, [tensor], '%s:' % name)

    with tf.name_scope('conv1_1') as scope:
        W_conv1_1 = weight_variable([3, 3, 3, 64])
        b_conv1_1 = bias_variable([64])
        h_conv1_1 = tf.nn.relu(conv2d(x_image, W_conv1_1) + b_conv1_1)
        h_conv1_1 = tf_print(h_conv1_1, 'h_conv1_1')

    with tf.name_scope('conv1_2') as scope:
        W_conv1_2 = weight_variable([3, 3, 64, 64])
        b_conv1_2 = bias_variable([64])
        h_conv1_2 = tf.nn.relu(conv2d(h_conv1_1, W_conv1_2) + b_conv1_2)
        h_conv1_2 = tf_print(h_conv1_2, 'h_conv1_2')

    with tf.name_scope('pool1') as scope:
        h_pool1 = max_pool_2x2(h_conv1_2)
        h_pool1 = tf_print(h_pool1, 'h_pool1')

    with tf.name_scope('conv2_1') as scope:
        W_conv2_1 = weight_variable([3, 3, 64, 128])
        b_conv2_1 = bias_variable([128])
        h_conv2_1 = tf.nn.relu(conv2d(h_pool1, W_conv2_1) + b_conv2_1)
        h_conv2_1 = tf_print(h_conv2_1, 'h_conv2_1')

    with tf.name_scope('conv2_2') as scope:
        W_conv2_2 = weight_variable([3, 3, 128, 128])
        b_conv2_2 = bias_variable([128])
        h_conv2_2 = tf.nn.relu(conv2d(h_conv2_1, W_conv2_2) + b_conv2_2)
        h_conv2_2 = tf_print(h_conv2_2, 'h_conv2_2')

    with tf.name_scope('pool2') as scope:
        h_pool2 = max_pool_2x2(h_conv2_2)
        h_pool2 = tf_print(h_pool2, 'h_pool2')

    with tf.name_scope('conv3_1') as scope:
        W_conv3_1 = weight_variable([3, 3, 128, 256])
        b_conv3_1 = bias_variable([256])
        h_conv3_1 = tf.nn.relu(conv2d(h_pool2, W_conv3_1) + b_conv3_1)
        h_conv3_1 = tf_print(h_conv3_1, 'h_conv3_1')

    with tf.name_scope('conv3_2') as scope:
        W_conv3_2 = weight_variable([3, 3, 256, 256])
        b_conv3_2 = bias_variable([256])
        h_conv3_2 = tf.nn.relu(conv2d(h_conv3_1, W_conv3_2) + b_conv3_2)
        h_conv3_2 = tf_print(h_conv3_2, 'h_conv3_2')

    with tf.name_scope('conv3_3') as scope:
        W_conv3_3 = weight_variable([3, 3, 256, 256])
        b_conv3_3 = bias_variable([256])
        h_conv3_3 = tf.nn.relu(conv2d(h_conv3_2, W_conv3_3) + b_conv3_3)
        h_conv3_3 = tf_print(h_conv3_3, 'h_conv3_3')

    with tf.name_scope('pool3') as scope:
        h_pool3 = max_pool_2x2(h_conv3_3)
        h_pool3 = tf_print(h_pool3, 'h_pool3')

    with tf.name_scope('conv4_1') as scope:
        W_conv4_1 = weight_variable([3, 3, 256, 512])
        b_conv4_1 = bias_variable([512])
        h_conv4_1 = tf.nn.relu(conv2d(h_pool3, W_conv4_1) + b_conv4_1)
        h_conv4_1 = tf_print(h_conv4_1, 'h_conv4_1')

    with tf.name_scope('conv4_2') as scope:
        W_conv4_2 = weight_variable([3, 3, 512, 512])
        b_conv4_2 = bias_variable([512])
        h_conv4_2 = tf.nn.relu(conv2d(h_conv4_1, W_conv4_2) + b_conv4_2)
        h_conv4_2 = tf_print(h_conv4_2, 'h_conv4_2')

    with tf.name_scope('conv4_3') as scope:
        W_conv4_3 = weight_variable([3, 3, 512, 512])
        b_conv4_3 = bias_variable([512])
        h_conv4_3 = tf.nn.relu(conv2d(h_conv4_2, W_conv4_3) + b_conv4_3)
        h_conv4_3 = tf_print(h_conv4_3, 'h_conv4_3')

    with tf.name_scope('pool4') as scope:
        h_pool4 = max_pool_2x2(h_conv4_3)
        h_pool4 = tf_print(h_pool4, 'h_pool4')

    with tf.name_scope('conv5_1') as scope:
        W_conv5_1 = weight_variable([3, 3, 512, 512])
        b_conv5_1 = bias_variable([512])
        h_conv5_1 = tf.nn.relu(conv2d(h_pool4, W_conv5_1) + b_conv5_1)
        h_conv5_1 = tf_print(h_conv5_1, 'h_conv5_1')

    with tf.name_scope('conv5_2') as scope:
        W_conv5_2 = weight_variable([3, 3, 512, 512])
        b_conv5_2 = bias_variable([512])
        h_conv5_2 = tf.nn.relu(conv2d(h_conv5_1, W_conv5_2) + b_conv5_2)
        h_conv5_2 = tf_print(h_conv5_2, 'h_conv5_2')

    with tf.name_scope('conv5_3') as scope:
        W_conv5_3 = weight_variable([3, 3, 512, 512])
        b_conv5_3 = bias_variable([512])
        h_conv5_3 = tf.nn.relu(conv2d(h_conv5_2, W_conv5_3) + b_conv5_3)
        h_conv5_3 = tf_print(h_conv5_3, 'h_conv5_3')

    with tf.name_scope('pool5') as scope:
        h_pool5 = max_pool_2x2(h_conv5_3)
        h_pool5 = tf_print(h_pool5, 'h_pool5')

    with tf.name_scope('fc6') as scope:
        w = 3 # 96/2^5
        W_fc6 = weight_variable([w*w*512, 4096])
        b_fc6 = bias_variable([4096])
        h_pool5_flat = tf.reshape(h_pool5, [-1, w*w*512])
        h_fc6 = tf.nn.relu(tf.matmul(h_pool5_flat, W_fc6) + b_fc6)
        h_fc6 = tf_print(h_fc6, 'h_fc6')

        h_fc6 = tf.nn.dropout(h_fc6, keep_prob)
        h_fc6 = tf_print(h_fc6, 'h_fc6_drop')

    with tf.name_scope('fc7') as scope:
        W_fc7 = weight_variable([4096, 4096])
        b_fc7 = bias_variable([4096])
        h_fc7 = tf.nn.relu(tf.matmul(h_fc6, W_fc7) + b_fc7)
        h_fc7 = tf_print(h_fc7, 'h_fc7')

        h_fc7 = tf.nn.dropout(h_fc7, keep_prob)
        h_fc7 = tf_print(h_fc7, 'h_fc7_drop')

    with tf.name_scope('fc8') as scope:
        W_fc8 = weight_variable([4096, NUM_CLASSES])
        b_fc8 = bias_variable([NUM_CLASSES])
        b_fc8 = tf_print(b_fc8, 'b_fc8')

    with tf.name_scope('softmax') as scope:
        y_conv = tf.nn.softmax(tf.matmul(h_fc7, W_fc8) + b_fc8)
        y_conv = tf_print(y_conv, 'y_conv')

    return y_conv

def loss(logits, labels):
    print 'loss:', logits, labels

    # 交差エントロピーの計算
    cross_entropy = -tf.reduce_sum(labels*tf.log(logits))
    # TensorBoardで表示するよう指定
    tf.scalar_summary("cross_entropy", cross_entropy)
    return cross_entropy

def training(loss, learning_rate):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_step

def accuracy(logits, labels):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.scalar_summary("accuracy", accuracy)
    return accuracy

def show(img):
    return
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():

    with tf.Graph().as_default():
        images, labels, filename = momo_input.load_data([FLAGS.train], FLAGS.batch_size)
        print 'start', images, labels

        keep_prob = tf.placeholder("float")
        global_step = tf.placeholder("float")

        logits = inference(images)
        print 'make_graph images:%s, logits:%s, labels:%s' % (images, logits, labels)

        loss_value = loss(logits, labels)

        train_op = training(loss_value, FLAGS.learning_rate)

        acc = accuracy(logits, labels)

        saver = tf.train.Saver(max_to_keep = FLAGS.max_steps)
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        tf.train.start_queue_runners(sess)
        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph_def)

        for step in xrange(FLAGS.max_steps):

            start_time = time.time()
            _, loss_result, acc_res = sess.run([train_op, loss_value, acc], feed_dict={keep_prob: 0.5, global_step: step})
            duration = time.time() - start_time

            if step % 10 == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                'sec/batch)')
                print (format_str % (datetime.now(), step, loss_result,
                examples_per_sec, sec_per_batch))

                print 'acc_res', acc_res

            if step % 100 == 0:
                summary_str = sess.run(summary_op,feed_dict={
                    keep_prob: 1.0})
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 100 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

if __name__ == '__main__':
    main()

