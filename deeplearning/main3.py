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
import momo_input

NUM_CLASSES = 5
#IMAGE_SIZE = 112
#IMAGE_SIZE = 56
IMAGE_SIZE = 28
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
flags.DEFINE_integer('max_steps', 1000, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 30, 'Batch size Must divide evenly into the dataset sizes.')
flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')
#flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')

def inference(x_image, keep_prob):

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

    # 入力をIMAGE_SIZEx IMAGE_SIZE x3に変形
    #x_image = tf.reshape(images, [-1, IMAGE_SIZE, IMAGE_SIZE, 3])
    x_image = tf_print(x_image, 'x_image:')

    # 畳み込み層1の作成
    with tf.name_scope('conv1') as scope:
        W_conv1 = weight_variable([5, 5, 3, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_conv1 = tf_print(h_conv1, 'h_conv1')

    # プーリング層1の作成
    with tf.name_scope('pool1') as scope:
        h_pool1 = max_pool_2x2(h_conv1)
        h_pool1 = tf_print(h_pool1, 'h_pool1')

    # 畳み込み層2の作成
    with tf.name_scope('conv2') as scope:
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_conv2 = tf_print(h_conv2, 'h_conv2')

    # プーリング層2の作成
    with tf.name_scope('pool2') as scope:
        h_pool2 = max_pool_2x2(h_conv2)
        h_pool2 = tf_print(h_pool2, 'h_pool2')

    # 全結合層1の作成
    with tf.name_scope('fc1') as scope:
        w = IMAGE_SIZE / 4
        W_fc1 = weight_variable([w*w*64, 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, w*w*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        h_fc1 = tf_print(h_fc1, 'h_fc1')

        # dropoutの設定
        # 途中でNaNになってvalidationにひっかかるのでコメントアウト
        #h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        #h_fc1_drop = tf_print(h_fc1_drop, 'h_fc1_drop')
        h_fc1_drop = h_fc1

    # 全結合層2の作成
    with tf.name_scope('fc2') as scope:
        W_fc2 = weight_variable([1024, NUM_CLASSES])
        b_fc2 = bias_variable([NUM_CLASSES])
        b_fc2 = tf_print(b_fc2, 'b_fc2')

    # ソフトマックス関数による正規化
    with tf.name_scope('softmax') as scope:
        y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        y_conv = tf_print(y_conv, 'y_conv')

    # 各ラベルの確率のようなものを返す
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

        logits = inference(images, keep_prob)
        print 'make_graph images:%s, logits:%s, labels:%s' % (images, logits, labels)

        loss_value = loss(logits, labels)

        train_op = training(loss_value, FLAGS.learning_rate)

        acc = accuracy(logits, labels)

        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        tf.train.start_queue_runners(sess)
        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph_def)

        for step in xrange(FLAGS.max_steps):

            for i in range(716/FLAGS.batch_size):
                sess.run([train_op, filename], feed_dict={keep_prob: 0.5})

            start_time = time.time()
            _, loss_result, acc_res = sess.run([train_op, loss_value, acc], feed_dict={keep_prob: 0.5})
            duration = time.time() - start_time
            #assert not np.isnan(loss_result), 'Model diverged with loss = NaN'

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

