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

def inference_deep2(x_image, keep_prob, image_size, num_classes):

    with tf.name_scope('conv1_1') as scope:
        W_conv1_1 = weight_variable([3, 3, 3, 32])
        b_conv1_1 = bias_variable([32])
        h_conv1_1 = tf.nn.relu(conv2d(x_image, W_conv1_1) + b_conv1_1)
        print h_conv1_1

    with tf.name_scope('pool1') as scope:
        h_pool1 = max_pool_2x2(h_conv1_1)
        print h_pool1

    with tf.name_scope('conv2_1') as scope:
        W_conv2_1 = weight_variable([3, 3, 32, 64])
        b_conv2_1 = bias_variable([64])
        h_conv2_1 = tf.nn.relu(conv2d(h_pool1, W_conv2_1) + b_conv2_1)
        print h_conv2_1

    with tf.name_scope('pool2') as scope:
        h_pool2 = max_pool_2x2(h_conv2_1)
        print h_pool2

    with tf.name_scope('conv3_1') as scope:
        W_conv3_1 = weight_variable([3, 3, 64, 128])
        b_conv3_1 = bias_variable([128])
        h_conv3_1 = tf.nn.relu(conv2d(h_pool2, W_conv3_1) + b_conv3_1)
        print h_conv3_1

    with tf.name_scope('pool3') as scope:
        h_pool3 = max_pool_2x2(h_conv3_1)
        print h_pool3

    with tf.name_scope('conv4_1') as scope:
        W_conv4_1 = weight_variable([3, 3, 128, 256])
        b_conv4_1 = bias_variable([256])
        h_conv4_1 = tf.nn.relu(conv2d(h_pool3, W_conv4_1) + b_conv4_1)
        print h_conv4_1

    with tf.name_scope('pool4') as scope:
        h_pool4 = max_pool_2x2(h_conv4_1)
        print h_pool4

    with tf.name_scope('fc6') as scope:
        w = image_size / pow(2,4)
        W_fc6 = weight_variable([w*w*256, 1024])
        print W_fc6.get_shape()
        b_fc6 = bias_variable([1024])
        h_pool4_flat = tf.reshape(h_pool4, [-1, w*w*256])
        h_fc6 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc6) + b_fc6)
        print h_fc6

        h_fc6 = tf.nn.dropout(h_fc6, keep_prob)
        print h_fc6

    with tf.name_scope('fc7') as scope:
        W_fc7 = weight_variable([1024, 256])
        b_fc7 = bias_variable([256])
        h_fc7 = tf.nn.relu(tf.matmul(h_fc6, W_fc7) + b_fc7)
        print h_fc7

        h_fc7 = tf.nn.dropout(h_fc7, keep_prob)
        print h_fc7

    with tf.name_scope('fc8') as scope:
        W_fc8 = weight_variable([256, num_classes])
        b_fc8 = bias_variable([num_classes])
        print b_fc8

    with tf.name_scope('softmax') as scope:
        y_conv = tf.nn.softmax(tf.matmul(h_fc7, W_fc8) + b_fc8)
        print y_conv

    return y_conv


def inference_deep(x_image, keep_prob, image_size, num_classes):

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
        w = image_size / pow(2,5) #96/2^5=3
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
        W_fc8 = weight_variable([4096, num_classes])
        b_fc8 = bias_variable([num_classes])
        b_fc8 = tf_print(b_fc8, 'b_fc8')

    with tf.name_scope('softmax') as scope:
        y_conv = tf.nn.softmax(tf.matmul(h_fc7, W_fc8) + b_fc8)
        y_conv = tf_print(y_conv, 'y_conv')

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
