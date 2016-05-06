#!/usr/bin/env python
#! -*- coding: utf-8 -*-

import sys
import numpy as np
import tensorflow as tf
import cv2
import os

NUM_CLASSES = 5
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE*IMAGE_SIZE*3

W_conv1 = None
h_conv1 = None
h_conv2 = None
h_pool2 = None

# [1,28,28,32]
def dump_hconv(ary, outdir = 'out_hconv1'):
    if not os.path.exists(outdir):
      os.mkdir(outdir)

    #print ary.shape
    t0 = ary.transpose()
    for i, t1 in enumerate(t0):
      path = os.path.join(outdir, '%s.png' % i)
      #print t1
      t2 = [n*255 for n in t1]
      t3 = np.array(t2)
      h,w,ch = t3.shape
      M = cv2.getRotationMatrix2D((w/2,h/2),-90,1)
      t4 = cv2.warpAffine(t3, M, (w,h))
      #print t2
      cv2.imwrite(path, t4)
    return

    #ary = (5,5,3,32)
    t = ary.transpose() # => (32,3,5,5)
    for i, t0 in enumerate(t):
      path = os.path.join(outdir, '%s.png' % i)
      t1 = t0.transpose() # => (5,5,3)
      #print 't1:%s' % t1[0][0]
      t2 = map(lambda n:(map(lambda m:sum(m), n)), t1)
      #print 't2:%s' % t2[0][0]
      t4 = map(lambda n:(map(lambda m:nega_posi_to_rgb(m), n)), t2)
      #print 't4:%s' % t4[0][0]
      cv2.imwrite(path, np.array(t4))
      #print path


def dump_wconv(ary, outdir):
    return

    # RGBの合計値を出力
    if not os.path.exists(outdir):
      os.mkdir(outdir)

    #ary = (5,5,3,32)
    t = ary.transpose() # => (32,3,5,5)
    for i, t0 in enumerate(t):
      path = os.path.join(outdir, '%s.png' % i)
      t1 = t0.transpose() # => (5,5,3)
      #print 't1:%s' % t1[0][0]
      t2 = map(lambda n:(map(lambda m:sum(m), n)), t1)
      #print 't2:%s' % t2[0][0]
      t4 = map(lambda n:(map(lambda m:nega_posi_to_rgb(m), n)), t2)
      #print 't4:%s' % t4[0][0]
      cv2.imwrite(path, np.array(t4))
      #print path

def dump(ary):
    # RGBのBのfilterだけをimgとして出力
    outdir = 'out_filter'
    if not os.path.exists(outdir):
      os.mkdir(outdir)

    #ary = (5,5,3,32)
    t = ary.transpose() # => (32,3,5,5)
    for i, t0 in enumerate(t):
      path = os.path.join(outdir, '%s.png' % i)
      t1 = t0.transpose() # => (5,5,3)
      t2 = t1.transpose() # => (3,5,5)
      t3 = t2[0]
      #print t3
      # [-1..0..+1] を [blue...black....red] にマッピング
      t4 = map(lambda n:(map(lambda m:nega_posi_to_rgb(m), n)), t3)
      #print t4
      cv2.imwrite(path, np.array(t4))
      #print path

def nega_posi_to_rgb(n):
  seed = [0.0, 0.0, 0.0]
  if n > 0:
    seed[0] += n
  else:
    seed[2] += (-1 * n)
  return [int(seed[0]*255), int(seed[1]*255), int(seed[2]*255)]

def imgshow(img):
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def inference(images_placeholder, keep_prob):

    def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

    def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

    def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')

    # [batch_size = -1, width, height, channel=3]
    x_image = tf.reshape(images_placeholder, [-1, IMAGE_SIZE, IMAGE_SIZE, 3])
    # 28x28,3ch image

    # 畳み込み層
    with tf.name_scope('conv1') as scope:
        # [width, height, channels, output_channels=32])
        global W_conv1
        global h_conv1
        W_conv1 = weight_variable([5, 5, 3, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        #28x28,32ch image

    # プーリング層
    with tf.name_scope('pool1') as scope:
        h_pool1 = max_pool_2x2(h_conv1)
        #14x14,32ch image

    with tf.name_scope('conv2') as scope:
        global h_conv2
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        #14x14,64ch image

    with tf.name_scope('pool2') as scope:
        global h_pool2
        h_pool2 = max_pool_2x2(h_conv2)
        #7x7,64ch image

    # 全結合層
    with tf.name_scope('fc1') as scope:
        w = IMAGE_SIZE / 4
        W_fc1 = weight_variable([w * w * 64, 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, w * w * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        # input: 7*7*64 input
        # hidden: 1024 unit

    with tf.name_scope('fc2') as scope:
        W_fc2 = weight_variable([1024, NUM_CLASSES])
        b_fc2 = bias_variable([NUM_CLASSES])
        # 5 unit中間層

    with tf.name_scope('softmax') as scope:
        y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    return y_conv

if __name__ == '__main__':
    img_files = []
    test_image = []

    for i in range(1, len(sys.argv)):
        filepath = sys.argv[i]
        if not os.path.isfile(filepath):
          continue
        img_files.append(filepath)
        img = cv2.imread(filepath)
        img = cv2.resize(img, (28, 28))
        test_image.append(img.flatten().astype(np.float32)/255.0)

    test_image = np.asarray(test_image)
    images_placeholder = tf.placeholder("float", shape=(None, IMAGE_PIXELS))
    keep_prob = tf.placeholder("float")

    logits = inference(images_placeholder, keep_prob)

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    saver.restore(sess, "model.ckpt")

    dump_wconv(sess.run(W_conv1), 'out_rgb_mix')
    #(5, 5, 3, 32)

    names = {
      0: "reni",
      1: "kanako",
      2: "shiori",
      3: "arin",
      4: "momoka",
    }

    for i, image in enumerate(test_image):
        print img_files[i]

        h_conv1_res = h_conv1.eval(feed_dict={images_placeholder: [image]})
        dump_hconv(h_conv1_res)

        h_conv2_res = h_conv2.eval(feed_dict={images_placeholder: [image]})
        dump_hconv(h_conv2_res, 'out_hconv2')

        h_pool2_res = h_pool2.eval(feed_dict={images_placeholder: [image]})
        dump_hconv(h_pool2_res, 'out_hpool2')

        softmax = logits.eval(feed_dict={images_placeholder: [image], keep_prob: 1.0 })
        #print softmax.shape
        print [round(n * 100.0, 1) for n in softmax[0]]

        pred = np.argmax(softmax[0])
        print names[pred]


