#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

def inference(images_placeholder, keep_prob, image_size, num_classes):

    x_image = tf.reshape(images_placeholder, [-1, image_size, image_size, 3])
    print x_image

    with tf.name_scope('conv1') as scope:
        W_conv1 = weight_variable([5, 5, 3, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        print h_conv1

    with tf.name_scope('pool1') as scope:
        h_pool1 = max_pool_2x2(h_conv1)
        print h_pool1

    with tf.name_scope('conv2') as scope:
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        print h_conv2

    with tf.name_scope('pool2') as scope:
        h_pool2 = max_pool_2x2(h_conv2)
        print h_pool2

    with tf.name_scope('fc1') as scope:
        w = image_size / 4
        W_fc1 = weight_variable([w*w*64, 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, w*w*64])
        print h_pool2_flat
        h_fc1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
        h_fc1_drop = tf.nn.dropout(tf.nn.relu(h_fc1), keep_prob)
        print h_fc1_drop

    with tf.name_scope('fc2') as scope:
        W_fc2 = weight_variable([1024, num_classes])
        b_fc2 = bias_variable([num_classes])
        h_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        print h_fc2

    with tf.name_scope('softmax') as scope:
        y_conv=tf.nn.softmax(h_fc2)
        print y_conv

    return y_conv

def inference_deep(images_placeholder, keep_prob, image_size, num_classes):

    x_image = tf.reshape(images_placeholder, [-1, image_size, image_size, 3])
    print x_image

    with tf.name_scope('conv1') as scope:
        W_conv1 = weight_variable([3, 3, 3, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        print h_conv1

    with tf.name_scope('pool1') as scope:
        h_pool1 = max_pool_2x2(h_conv1)
        print h_pool1

    with tf.name_scope('conv2') as scope:
        W_conv2 = weight_variable([3, 3, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        print h_conv2

    with tf.name_scope('pool2') as scope:
        h_pool2 = max_pool_2x2(h_conv2)
        print h_pool2

    with tf.name_scope('conv3') as scope:
        W_conv3 = weight_variable([3, 3, 64, 128])
        b_conv3 = bias_variable([128])
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
        print h_conv3

    with tf.name_scope('pool3') as scope:
        h_pool3 = max_pool_2x2(h_conv3)
        print h_pool3

    with tf.name_scope('fc1') as scope:
        w = image_size / pow(2,3)
        W_fc1 = weight_variable([w*w*128, 1024])
        b_fc1 = bias_variable([1024])
        h_pool3_flat = tf.reshape(h_pool3, [-1, w*w*128])
        print h_pool3_flat
        h_fc1 = tf.matmul(h_pool3_flat, W_fc1) + b_fc1
        h_fc1_drop = tf.nn.dropout(tf.nn.relu(h_fc1), keep_prob)
        print h_fc1_drop

    with tf.name_scope('fc2') as scope:
        W_fc2 = weight_variable([1024, num_classes])
        b_fc2 = bias_variable([num_classes])
        h_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        print h_fc2

    with tf.name_scope('softmax') as scope:
        y_conv=tf.nn.softmax(h_fc2)
        print y_conv

    return y_conv

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def tf_print(tensor, name):
    print name, tensor
    return tensor

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def loss(logits, labels):
    #http://qiita.com/ikki8412/items/3846697668fc37e3b7e0
    #cross_entropy = -tf.reduce_sum(labels*tf.log(logits))
    cross_entropy = -tf.reduce_sum(labels*tf.log(tf.clip_by_value(logits,1e-10,1.0)))
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
