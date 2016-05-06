#!/usr/bin/env python
#! -*- coding: utf-8 -*-

import sys
import numpy as np
import tensorflow as tf
import cv2
import os
import operator
import scipy.ndimage
import mcz_input
import mcz_model

def main():
    img_files = []
    test_image = []

    for i in range(1, len(sys.argv)):
        filepath = sys.argv[i]
        if not os.path.isfile(filepath):
          continue
        img_files.append(filepath)
        img = cv2.imread(filepath)
        img = cv2.resize(img, (28, 28))
        dump_image(img, 'out_0_img28')
        test_image.append(img.flatten().astype(np.float32)/255.0)

    test_image = np.asarray(test_image)
    pixel_size = mcz_model.IMAGE_SIZE * mcz_model.IMAGE_SIZE * 3
    images_placeholder = tf.placeholder("float", shape=(None, pixel_size))
    keep_prob = tf.placeholder("float")

    #logits, w_fc2, b_fc2, h_fc2, w_fc1, b_fc1, h_fc1, h_pool2, w_conv2 = mcz_model.inference(images_placeholder, keep_prob) #for debug
    logits = mcz_model.inference(images_placeholder, keep_prob)

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    saver.restore(sess, "model.ckpt")

    for i, image in enumerate(test_image):
        print img_files[i]
        feed_dict = {images_placeholder: [image], keep_prob: 1.0 }

        softmax = logits.eval(feed_dict = feed_dict)
        print [round(n * 100.0, 1) for n in softmax[0]]

        pred = np.argmax(softmax[0])
        print mcz_input.MEMBER_NAMES[pred]

        continue
        #======== debug ===========
        #reni=0が最優先(それ以降は何でも良し)
        #rank_std = [4,0,1,2,3] #momoka
        rank_std = [0,1,2,3,4] #reni

        fc2_rank = get_fc_rank(w_fc2, b_fc2, h_fc2, rank_std, feed_dict)
        print 'fc2_rank',fc2_rank[0]
        #for i, idx in enumerate(fc2_rank):
        #    print 'fc2_rank', i, idx

        fc1_rank = get_fc_rank(w_fc1, b_fc1, h_fc1, fc2_rank, feed_dict)
        print 'fc1_rank',fc1_rank[0]
        #for i, idx in enumerate(fc1_rank):
        #    print 'fc1_rank', i, idx

        fc1_rank_p = [[idx, 1.0 - float(i) / float(len(fc1_rank))]  for i, idx in enumerate(fc1_rank)]
        fc1_rank_ps = [x[1] for x in sorted(fc1_rank_p, key=operator.itemgetter(0))]
        print 'fc1_rank_ps',fc1_rank_ps[0]
        #for i, f in enumerate(fc1_rank_ps):
        #    print i, f

        h_pool2_res = h_pool2.eval(feed_dict=feed_dict)
        _, w, h, ch = h_pool2_res.shape #1,7,7,64
        tmp = np.array(fc1_rank_ps)
        print tmp.shape
        h_pool2_rank = np.reshape(tmp, (ch,w,h))
        print h_pool2_rank.shape

        h_conv2 = np.array([scipy.ndimage.zoom(hp2r, 2, order=0) for hp2r in h_pool2_rank]) #64,14,14
        print 'h_conv2',h_conv2.shape
        #print h_conv2[0]
        h_conv2_ch, hc2w, hc2h = h_conv2.shape

        #for i, hconv2item in enumerate(h_conv2):
            #print hconv2item
            #hconv2item = hconv2item * 255
            #cv2.imwrite('hconv2item_%s.jpg' % i, hconv2item)
        hconv2item_mean = np.mean(h_conv2, axis=0) * 255.0
        cv2.imwrite('hconv2item_mean.jpg', hconv2item_mean)
        print 'hconv2item_mean',hconv2item_mean

        w_conv2_res = w_conv2.eval().transpose()
        print 'w_conv2_res', w_conv2_res.shape #(64, 32, 5, 5)
        wc2out, wc2in, wcw, wch  = w_conv2_res.shape

        #(32,14,14)
        h_conv1 = np.zeros((wc2in, hc2w, hc2h), dtype="float")
        print h_conv1.shape

        for (hc2, wcv2) in zip(h_conv2, w_conv2_res): #64time
            #print hc2.shape #(14,14)
            #print wcv2.shape #(32,5,5)
            for i, wcv2e in enumerate(wcv2): #32time , (5,5)
                conved = conv(hc2, wcv2e) #(14,14)
                h_conv1[i] = h_conv1[i] + conved

        h1i_max = 0
        h1i_min = 0
        print 'hconv1item'
        for i, hconv1item in enumerate(h_conv1):
            fl = hconv1item.flatten()
            #print 'fl',fl,min(fl)
            #flmin = min(fl)
            #print 'min', flmin, 'min_idx',np.where(fl==flmin)
            #print 'max', max(fl)
            #h1i_max = max([h1i_max, max()])
            #h1i_min = min([h1i_min, min(hconv1item.flatten())])
            #cv2.imwrite('hconv1item_%s.jpg' % i, hconv1item)
        print h1i_max, h1i_min # 0.991607962861 -0.841329923029 (-1 ~ +1)
        #print h_conv1[0]

        h_conv1_p1 = h_conv1 + 1.0 #(0~+2)
        h_conv1_p1_d2 = h_conv1_p1 / 2.0 * 255.0 #(0~+255)
        for i, hcpd in enumerate(h_conv1_p1_d2):
            cv2.imwrite('hcpd_%s.jpg' % i, hcpd)

        #print 'h_conv1_p1_d2'
        #print h_conv1_p1_d2[0]

        return

        alist = []
        for i,hc1 in enumerate(h_conv1):
            print hc1
            for y,rows in enumerate(aaa):
                for x,val in enumerate(rows):
                    if val > 0:
                        aaa[y][x] = 255#val
                        alist.append(val)
                    else:
                        aaa[y][x] = 0
            print 'max',max(alist)
            print 'min',min(alist)
            cv2.imwrite('aaa_%s.jpg' % i, aaa)

def get_fc_rank(w_fc2, b_fc2, h_fc2, rank_std, feed_dict):
    b_fc2_res = b_fc2.eval()
    h_fc2_res = h_fc2.eval(feed_dict=feed_dict)

    print 'w:', w_fc2.eval().shape
    print 'b:', b_fc2_res.shape
    print 'h:', h_fc2_res.shape

    bh_fc2 = h_fc2_res[0] - b_fc2_res
    w_fc2_res = w_fc2.eval().tolist()

    w_fc2_res_sort_by_val = sorted(w_fc2_res, key=operator.itemgetter(rank_std[0], rank_std[1], rank_std[2], rank_std[3], rank_std[4]))
    fc2_rank = [w_fc2_res.index(val) for val in w_fc2_res_sort_by_val]

    return fc2_rank

def tf_print(tensor, name):
    print name, tensor
    return tf.Print(tensor, [tensor], '%s:' % name)

def rotate(img):
    img = np.fliplr(img)
    h,w,ch = img.shape
    rotate = np.zeros((28,28,3))
    for y, rows in enumerate(img):
        for x, col in enumerate(rows):
            rotate[27-x][y] = col
    return rotate

# imgを90度回転&左右反転して書き出し
def dump_image(img, outdir = 'out_img'):
    if not os.path.exists(outdir):
      os.mkdir(outdir)

    # original
    org_path = os.path.join(outdir, 'org_img.png')
    cv2.imwrite(org_path, img)

    # rotate & split by channel
    img = rotate(img)
    rgb = cv2.split(img)
    cv2.imwrite(os.path.join(outdir, 'img_r.png'), rgb[0])
    cv2.imwrite(os.path.join(outdir, 'img_g.png'), rgb[1])
    cv2.imwrite(os.path.join(outdir, 'img_b.png'), rgb[2])

# [1,28,28,32]
def dump_hconv(ary, outdir = 'out_hconv1'):
    if not os.path.exists(outdir):
      os.mkdir(outdir)

    t0 = ary.transpose()
    for i, t1 in enumerate(t0):
      path = os.path.join(outdir, '%s.png' % i)
      t2 = [n*255 for n in t1]
      t3 = np.array(t2)
      cv2.imwrite(path, t3)

def dump_wconv_(ary, outdir):
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
      #t5 = np.fliplr(t4)
      cv2.imwrite(path, np.array(t4))
      #print path

def dump_wconv(ary, outdir):
    # RGBの合計値を出力
    if not os.path.exists(outdir):
      os.mkdir(outdir)

    #ary = (5,5,3,32)
    t = ary.transpose() # => (32,3,5,5)
    for i, t0 in enumerate(t):
        for j, t1 in enumerate(t0):
            t2 = map(lambda n:(map(lambda m:nega_posi_to_rgb(m), n)), t1)
            #[5,5]
            path = os.path.join(outdir, '%s_%s.png' % (i, j))
            cv2.imwrite(path, np.array(t2))
            """
            #t1 = t0.transpose() # => (5,5,3)
            #print 't1:%s' % t1[0][0]
            t2 = map(lambda n:(map(lambda m:sum(m), n)), t1)
            #print 't2:%s' % t2[0][0]
            t4 = map(lambda n:(map(lambda m:nega_posi_to_rgb(m), n)), t2)
            #print 't4:%s' % t4[0][0]
            #t5 = np.fliplr(t4)
            cv2.imwrite(path, np.array(t4))
            #print path
            """

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


def test():
    a = np.array([
        [1,1,1],
        [1,1,1],
        [1,1,1]], dtype="float")
    """
    [1,1,1,1,1],
    [1,1,1,1,1],
    [1,1,1,1,1],
    [1,1,1,1,1],
    [1,1,1,1,1]], dtype="float")
    """
    print a

    w = np.array([
        [0.5, 0.5, 0.5],
        [0.5, 2.0, 0.5],
        [0.5, 0.5, 0.5]], dtype="float")
    print w

    print conv(a, w)

def conv(a, w):
    aw,_ = a.shape
    ww,_ = w.shape
    ad = np.zeros((aw+2,aw+2), dtype="float")
    res = np.zeros((aw,aw), dtype="float")
    adw,_ = ad.shape
    for i in range(1, aw+1):
        for j in range(1, aw+1):
            ad[i][j] = a[i-1][j-1]

    for i in range(0, adw - ww):
        for j in range(0, adw - ww):
            for k in range(0, ww):
                for l in range(0, ww):
                    ad[i+k][j+l] = ad[i+k][j+l] * w[k][l]
    for i in range(1, aw+1):
        for j in range(1, aw+1):
            res[i-1][j-1] = ad[i][j]

    return res

if __name__ == '__main__':
    main()
    #test()
