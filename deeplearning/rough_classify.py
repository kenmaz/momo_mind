#!/usr/bin/env python
#! -*- coding: utf-8 -*-

import sys
import numpy as np
import tensorflow as tf
import cv2
import os
from main import names
from main import inference
from datetime import datetime
import shutil

NUM_CLASSES = 5
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE * 3

def mk_outdirs():
    dirs = {}

    rootdir = 'out.%s' % datetime.now().strftime('%s')
    #print rootdir
    if not os.path.exists(rootdir):
        os.mkdir(rootdir)

    _names = names()
    for i in _names.keys():
        name = _names[i]
        outdir = '%s/%s/' % (rootdir, name)
        #print outdir
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        dirs[i] = outdir
    return dirs

def execute(outdirs, filepaths):
    test_image = []
    for filepath in filepaths:
        img = cv2.imread(filepath)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        test_image.append(img.flatten().astype(np.float32)/255.0)

    _names = names()
    images_placeholder = tf.placeholder("float", shape=(None, IMAGE_PIXELS))
    keep_prob = tf.placeholder("float")
    logits = inference(images_placeholder, keep_prob)

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    saver.restore(sess, "model.ckpt")

    for i, image in enumerate(test_image):
        softmax = logits.eval(feed_dict={images_placeholder: [image], keep_prob: 1.0 })
        print [round(n * 100.0, 1) for n in softmax[0]]
        pred = np.argmax(softmax[0])
        print _names[pred]
        outdir = outdirs[pred]
        filepath = filepaths[i]
        filename = os.path.basename(filepath)
        distpath = os.path.join(outdir, filename)
        print distpath
        shutil.copyfile(filepath, distpath)

if __name__ == '__main__':
    outdirs = mk_outdirs()

    outdir = sys.argv[1]
    if not os.path.isdir(outdir):
        sys.exit('%s is not directory' % outdir)
    exts = ['.PNG','.JPG','.JPEG']
    paths = []
    for dirpath, dirnames, filenames in os.walk(outdir):
        for filename in filenames:
            (fn,ext) = os.path.splitext(filename)
            if ext.upper() in exts:
                path = os.path.join(dirpath,filename)
                paths.append(path)
    execute(outdirs, paths)
