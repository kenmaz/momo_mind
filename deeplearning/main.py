#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.python.platform
from datetime import datetime
import variation
import random
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
#flags.DEFINE_string('train', 'train_1.txt', 'File name of train data')
flags.DEFINE_string('test', 'test.txt', 'File name of train data')
#flags.DEFINE_string('test', 'test_osaretai.txt', 'File name of train data')
flags.DEFINE_string('train_dir', LOGDIR, 'Directory to put the training data.')
#flags.DEFINE_integer('max_steps', 200, 'Number of steps to run trainer.')
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

    # 入力をIMAGE_SIZEx IMAGE_SIZE x3に変形
    #x_image = tf.reshape(images_placeholder, [-1, IMAGE_SIZE, IMAGE_SIZE, 3])

    # 畳み込み層1の作成
    with tf.name_scope('conv1') as scope:
        W_conv1 = weight_variable([5, 5, 3, 32])

        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        print 'h_conv1', h_conv1

    # プーリング層1の作成
    with tf.name_scope('pool1') as scope:
        h_pool1 = max_pool_2x2(h_conv1)

    # 畳み込み層2の作成
    with tf.name_scope('conv2') as scope:
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # プーリング層2の作成
    with tf.name_scope('pool2') as scope:
        h_pool2 = max_pool_2x2(h_conv2)
        print 'h_pool2', h_pool2

    # 全結合層1の作成
    with tf.name_scope('fc1') as scope:
        reshape = tf.reshape(h_pool2, [FLAGS.batch_size, -1])
        dim = reshape.get_shape()[1].value
        W_fc1 = weight_variable([dim, 1024])
        b_fc1 = bias_variable([1024])
        h_fc1 = tf.nn.relu(tf.matmul(reshape, W_fc1) + b_fc1)
        print 'h_fc1', h_fc1

        # dropoutの設定
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        print 'h_fc2_drop', h_fc1_drop

    # 全結合層2の作成
    with tf.name_scope('fc2') as scope:
        W_fc2 = weight_variable([1024, NUM_CLASSES])
        b_fc2 = bias_variable([NUM_CLASSES])
        print 'b_fc2', b_fc2

    # ソフトマックス関数による正規化
    with tf.name_scope('softmax') as scope:
        y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        print 'y_conv', y_conv

    # 各ラベルの確率のようなものを返す
    return y_conv

def loss(logits, labels):
    """ lossを計算する関数

    引数:
      logits: ロジットのtensor, float - [batch_size, NUM_CLASSES]
      labels: ラベルのtensor, int32 - [batch_size, NUM_CLASSES]

    返り値:
      cross_entropy: 交差エントロピーのtensor, float

    """
    print 'loss:', logits, labels

    # 交差エントロピーの計算
    cross_entropy = -tf.reduce_sum(labels*tf.log(logits))
    # TensorBoardで表示するよう指定
    tf.scalar_summary("cross_entropy", cross_entropy)
    return cross_entropy

def training(loss, learning_rate):
    """ 訓練のOpを定義する関数

    引数:
      loss: 損失のtensor, loss()の結果
      learning_rate: 学習係数

    返り値:
      train_step: 訓練のOp

    """

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_step

def accuracy(logits, labels):
    """ 正解率(accuracy)を計算する関数

    引数: 
      logits: inference()の結果
      labels: ラベルのtensor, int32 - [batch_size, NUM_CLASSES]

    返り値:
      accuracy: 正解率(float)

    """
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.scalar_summary("accuracy", accuracy)
    return accuracy


def names():
    names = {
      0: "reni",
      1: "kanako",
      2: "shiori",
      3: "arin",
      4: "momoka",
    }
    return names

def show(img):
    return
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def make_graph(images, labels, keep_prob):
    logits = inference(images, keep_prob)

    print 'make_graph images:%s, logits:%s, labels:%s' % (images, logits, labels)

    loss_value = loss(logits, labels)
    train_op = training(loss_value, FLAGS.learning_rate)
    acc = accuracy(logits, labels)
    return train_op, acc

def main():

    with tf.Graph().as_default():
        images, labels = momo_input.load_data([FLAGS.train], FLAGS.batch_size)
        print 'start', images, labels

        keep_prob = tf.placeholder("float")

        train_op, acc = make_graph(images, labels, keep_prob)

        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        tf.train.start_queue_runners(sess)
        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph_def)

        for step in range(FLAGS.max_steps):
            for i in range(716/FLAGS.batch_size):
                sess.run(train_op, feed_dict={
                  keep_prob: 0.5})

            train_accuracy = sess.run(acc, feed_dict={
                keep_prob: 1.0})
            print "step %d, training accuracy %g"%(step, train_accuracy)

            summary_str = sess.run(summary_op, feed_dict={
                keep_prob: 1.0})
            summary_writer.add_summary(summary_str, step)

        t_images, t_labels = load_data([FLAGS.train], FLAGS.batch_size)
        t_train_op, t_acc = make_graph(t_images, t_labels, keep_prob)
        print "test accuracy %g"%sess.run(t_acc, feed_dict={
            keep_prob: 1.0})

        save_path = saver.save(sess, "model.ckpt")

if __name__ == '__main__':
    main()

