#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.python.platform
import variation
import random

NUM_CLASS = 5
IMAGE_ORG_SIZE = 112
IMAGE_SIZE = 28
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 500

def show(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def load_data_for_test(csv, batch_size):
    return load_data(csv, batch_size, shuffle = False, distored = False)

def load_data(csv, batch_size, shuffle = True, distored = True):
    queue = tf.train.string_input_producer(csv, shuffle=shuffle)
    reader = tf.TextLineReader()
    key, value = reader.read(queue)
    filename, label = tf.decode_csv(value, [["path"],[1]], field_delim=" ")

    label = tf.cast(label, tf.int64)
    label = tf.one_hot(label, depth = NUM_CLASS, on_value = 1.0, off_value = 0.0, axis = -1)

    jpeg = tf.read_file(filename)
    image = tf.image.decode_jpeg(jpeg, channels=3)
    image = tf.reshape(image, [IMAGE_ORG_SIZE, IMAGE_ORG_SIZE, 3])
    image = tf.image.resize_images(image, IMAGE_SIZE, IMAGE_SIZE)

    if distored:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.2, upper=1.8)

    float_image = tf.image.per_image_whitening(image)

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)
    return _generate_image_and_label_batch(float_image, label, filename,
        min_queue_examples, batch_size,
        shuffle=shuffle)

def _generate_image_and_label_batch(image, label, filename, min_queue_examples,
                                    batch_size, shuffle):

    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 16
    capacity = min_queue_examples + 3 * batch_size

    if shuffle:
        images, label_batch, filename = tf.train.shuffle_batch(
            [image, label, filename],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            #capacity=capacity,
            capacity=50000,
            #min_after_dequeue=min_queue_examples)
            min_after_dequeue=716)
    else:
        images, label_batch, filename = tf.train.batch(
            [image, label, filename],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    tf.image_summary('images', images)

    labels = tf.reshape(label_batch, [batch_size, NUM_CLASS])
    return images, labels, filename

def main2():
    with tf.Graph().as_default():
        images, labels, filename = load_data(['train.txt'], 30)
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        tf.train.start_queue_runners(sess)
        res_names = sess.run(filename)
        #for name in res_names:
            #print name
        res_list = sess.run(images)
        for res in res_list:
            res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
            #show(res)
        res_label = sess.run(labels)
        for label in res_label:
            print label

if __name__ == '__main__':
    main2()

