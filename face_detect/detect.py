# vim:fileencoding=utf-8

import numpy as np
import cv2
import sys
import os
import math

IMAGE_SIZE = 112
INPUT_SIZE = 96

def show(img):
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def validate(img_file):
  print img_file
  xml_dir = '/usr/local/Cellar/opencv/2.4.12_2/share/OpenCV/haarcascades'
  face_cascade = cv2.CascadeClassifier(os.path.join(xml_dir, 'haarcascade_frontalface_alt2.xml'))

  img = cv2.imread(img_file)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  scale = 1.02
  faces = face_cascade.detectMultiScale(gray, scale, 5)

def detect_face_rotate(img_file):
    img = cv2.imread(img_file)
    rows, cols, colors = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hypot = int(math.hypot(rows, cols))
    frame = np.zeros((hypot, hypot), np.uint8)
    frame[int((hypot - rows) * 0.5):int((hypot + rows) * 0.5), int((hypot - cols) * 0.5):int((hypot + cols) * 0.5)] = gray

    xml_dir = '/usr/local/Cellar/opencv/2.4.12_2/share/OpenCV/haarcascades'
    face_cascade = cv2.CascadeClassifier(os.path.join(xml_dir, 'haarcascade_frontalface_alt2.xml'))

    for deg in range(-50, 51, 5):
        print('deg:%s' % deg)
        M = cv2.getRotationMatrix2D((hypot * 0.5, hypot * 0.5), -deg, 1.0)
        rotated = cv2.warpAffine(frame, M, (hypot, hypot))
        faces = face_cascade.detectMultiScale(rotated, 1.02, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(rotated, (x, y), (x + w, y + h), (0, 0, 0), 2)
        show(rotated)

def detect_face(img_file, outdir = 'out'):
  xml_dir = '/usr/local/Cellar/opencv/2.4.12_2/share/OpenCV/haarcascades'
  face_cascade = cv2.CascadeClassifier(os.path.join(xml_dir, 'haarcascade_frontalface_alt2.xml'))
  eye_cascade = cv2.CascadeClassifier(os.path.join(xml_dir, 'haarcascade_eye.xml'))
  mouth_cascade = cv2.CascadeClassifier(os.path.join(xml_dir, 'haarcascade_mcs_mouth.xml'))
  nose_cascade = cv2.CascadeClassifier(os.path.join(xml_dir, 'haarcascade_mcs_nose.xml'))

  exts = ['.JPG','.JPEG']
  filename = os.path.basename(os.path.normpath(img_file))
  (fn, ext) = os.path.splitext(filename)
  if not ext.upper() in exts:
    print 'skip:%s' % img_file
    return

  print '..read %s' % img_file
  img = cv2.imread(img_file)

  print '..gray'
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  print '..face'
  scale = 1.01
  faces = face_cascade.detectMultiScale(gray, scale, 5)

  debug = False

  res = []

  for i, (x,y,w,h) in enumerate(faces):
    print '[face] %s %s' % (i, (x,y,w,h))

    eyes_ok = False
    mouth_ok = False
    nose_ok = False

    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

    #faceの中心上部にあればok
    eyes = eye_cascade.detectMultiScale(roi_gray, scale)
    for (ex,ey,ew,eh) in eyes:
      if debug:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
      h_diff = h/2 - (ey+eh/2)
      if h_diff > 0:
        eyes_ok = True

    #faceの中心下部付近にあればOK
    mouthes = mouth_cascade.detectMultiScale(roi_gray, scale)
    for (mx,my,mw,mh) in mouthes:
      if debug:
        cv2.rectangle(roi_color,(mx,my),(mx+mw,my+mh),(0,0,255),2)
      w_diff = abs(w/2 - (mx+mw/2))
      h_diff = h/2 - (my+mh/2)
      if h_diff < 0 and w_diff < w/10:
        mouth_ok = True

    #faceの中心下部付近にあればOK
    noses = nose_cascade.detectMultiScale(roi_gray, scale)
    for (nx,ny,nw,nh) in noses:
        #cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(255,255,0),2)
        w_diff = abs(w/2 - (nx+nw/2))
        h_diff = h/2 - (ny+nh/2)
        if h_diff < 0 and w_diff < w/10:
          nose_ok = True

    if eyes_ok and mouth_ok and nose_ok:
      if debug:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)

      out_file = '%s/%s_%s.jpg' % (outdir, fn, i)
      print out_file

      #上下左右に10%ほど余分に切り出し(無理なら不要)
      roi_color = img[y:y+h, x:x+w]
      print roi_color.shape

      margin = int(h * 0.2)
      img_w, img_h, img_ch = img.shape
      print 'margin:%s, img_w:%s img_h:%s' % (margin, img_w, img_h)
      if y - margin < 0 or x - margin < 0 or y + margin > img_h or x + margin > img_w:
        print 'cannot make margin %s ' % out_file
      else:
        roi_color = img[y - margin : y + h + margin, x - margin: x + w + margin]

      w,h,ch = roi_color.shape
      if w < IMAGE_SIZE or h < IMAGE_SIZE:
        print 'too small: %s' % out_file
        continue

      roi_color = cv2.resize(roi_color, (IMAGE_SIZE, IMAGE_SIZE))
      cv2.imwrite(out_file, roi_color)
      print 'write: %s' % out_file

      res.append(out_file)
    else:
      print 'eye:%s mouth:%s nose:%s' % (eyes_ok, mouth_ok, nose_ok)
      #img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)

  if debug:
    filename = os.path.basename(os.path.normpath(img_file))
    out_file = 'out/%s' % filename
    cv2.imwrite(out_file, img)
    print out_file

  return res

def detect_face_frontfaces(img_file):
  #face_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/2.4.12_2/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml')
  #face_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/2.4.12_2/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml')
  #face_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/2.4.12_2/share/OpenCV/haarcascades/haarcascade_frontalface_alt_tree.xml')
  #face_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/2.4.12_2/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')

  outdir = "out_alttree"
  if not os.path.exists(outdir):
    os.mkdir(outdir)

  eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
  scale = 1.01

  img = cv2.imread(img_file)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray, scale, 5)

  for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

  out_file = "%s/%s.jpg" % (outdir, "{0:.2f}".format(scale))
  cv2.imwrite(out_file, img)
  print out_file


def detect(img_file, out_file):
    #face_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/2.4.12_2/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml')
    #face_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/2.4.12_2/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml')
    #face_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/2.4.12_2/share/OpenCV/haarcascades/haarcascade_frontalface_alt_tree.xml')
    #face_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/2.4.12_2/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    img = cv2.imread(img_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        print (x,y,w,h)

        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        show(img)
        cv2.imwrite(out_file, img)

def detect_all(dir_path):
  outdir = 'out'
  if not os.path.exists(outdir):
    os.mkdir(outdir)

  if os.path.exists(dir_path) and os.path.isdir(dir_path):
    print dir_path
    for dirpath, dirnames, filenames in os.walk(dir_path):
      for filename in filenames:
        if not filename.startswith('.'):
          input_path = os.path.join(dirpath, filename)
          output_path = os.path.join(outdir, filename)
          print input_path
          print output_path
          try:
            detect_face(input_path, output_path)
          except:
            print 'error'

if __name__ == "__main__":
  param = sys.argv
  #detect_all(param[1])
  detect_face(param[1])
  #detect_face_rotate(param[1])
  #validate(param[1])
