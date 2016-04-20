# vim:fileencoding=utf-8

import numpy as np
import cv2
import sys
import os

def validate(img_file):
  print img_file
  xml_dir = '/usr/local/Cellar/opencv/2.4.12_2/share/OpenCV/haarcascades'
  face_cascade = cv2.CascadeClassifier(os.path.join(xml_dir, 'haarcascade_frontalface_alt2.xml'))

  img = cv2.imread(img_file)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  scale = 1.02
  faces = face_cascade.detectMultiScale(gray, scale, 5)

def detect_face(img_file):
  xml_dir = '/usr/local/Cellar/opencv/2.4.12_2/share/OpenCV/haarcascades'
  face_cascade = cv2.CascadeClassifier(os.path.join(xml_dir, 'haarcascade_frontalface_alt2.xml'))
  eye_cascade = cv2.CascadeClassifier(os.path.join(xml_dir, 'haarcascade_eye.xml'))
  mouth_cascade = cv2.CascadeClassifier(os.path.join(xml_dir, 'haarcascade_mcs_mouth.xml'))
  nose_cascade = cv2.CascadeClassifier(os.path.join(xml_dir, 'haarcascade_mcs_nose.xml'))

  print '..read %s' % img_file
  img = cv2.imread(img_file)

  print '..gray'
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  print '..face'
  scale = 1.01
  faces = face_cascade.detectMultiScale(gray, scale, 5)

  debug = False

  for i, (x,y,w,h) in enumerate(faces):
    print '[face] %s %s' % (i, (x,y,w,h))

    eyes_ok = False
    mouth_ok = False
    nose_ok = True

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
    #noses = nose_cascade.detectMultiScale(roi_gray, scale)
    #for (nx,ny,nw,nh) in noses:
    #    #cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(255,255,0),2)
    #    w_diff = abs(w/2 - (nx+nw/2))
    #    h_diff = h/2 - (ny+nh/2)
    #    if h_diff < 0 and w_diff < w/10:
    #      nose_ok = True

    if eyes_ok and mouth_ok and nose_ok:
      if debug:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
      filename = os.path.basename(os.path.normpath(img_file))
      out_file = 'out/%s_%s' % (i, filename)
      print out_file
      roi_color = img[y:y+h, x:x+w]
      cv2.imwrite(out_file, roi_color)
    else:
      print 'eye:%s mouth:%s nose:%s' % (eyes_ok, mouth_ok, nose_ok)
      #img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)

  if debug:
    filename = os.path.basename(os.path.normpath(img_file))
    out_file = 'out/%s' % filename
    cv2.imwrite(out_file, img)
    print out_file

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

  cv2.imshow('img',img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
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
  #validate(param[1])
