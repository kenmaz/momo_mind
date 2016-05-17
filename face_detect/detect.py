# vim:fileencoding=utf-8

import numpy as np
import cv2
import sys
import os
import math

IMAGE_SIZE = 112
INPUT_SIZE = 96

xml_dir = '/usr/local/Cellar/opencv/2.4.12_2/share/OpenCV/haarcascades'
#xml_dir = 'haarcascades'
face_cascade = cv2.CascadeClassifier(os.path.join(xml_dir, 'haarcascade_frontalface_alt2.xml'))
eye_cascade = cv2.CascadeClassifier(os.path.join(xml_dir, 'haarcascade_eye.xml'))
mouth_cascade = cv2.CascadeClassifier(os.path.join(xml_dir, 'haarcascade_mcs_mouth.xml'))
nose_cascade = cv2.CascadeClassifier(os.path.join(xml_dir, 'haarcascade_mcs_nose.xml'))

def detect_face_rotate(img_file, out_dir = 'out'):
    filename = os.path.basename(os.path.normpath(img_file))
    (fn, ext) = os.path.splitext(filename)
    img_dir = '%s/%s' % (out_dir, fn)
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)

    img = cv2.imread(img_file)
    rows, cols, colors = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hypot = int(math.hypot(rows, cols))
    frame = np.zeros((hypot, hypot), np.uint8)
    frame[int((hypot - rows) * 0.5):int((hypot + rows) * 0.5), int((hypot - cols) * 0.5):int((hypot + cols) * 0.5)] = gray
    results = []
    face_id_seed = 0

    #5度ずつ元画像を回転し、顔の候補を全部取得
    for deg in range(-50, 51, 5):
        print('deg:%s' % deg)
        M = cv2.getRotationMatrix2D((hypot * 0.5, hypot * 0.5), -deg, 1.0)
        rotated = cv2.warpAffine(frame, M, (hypot, hypot))
        #"""
        out_file = '%s/deg_%s.jpg' % (img_dir, deg)
        cv2.imwrite(out_file, rotated)
        #"""

        faces = face_cascade.detectMultiScale(rotated, 1.02, 5)
        for (x, y, w, h) in faces:
            face_cand = rotated[y:y+h, x:x+w]

            center = (int(x + w * 0.5), int(y + h * 0.5))
            origin = (int(hypot * 0.5), int(hypot * 0.5))
            r_deg = -deg
            center_org = rotate_coord(center, origin, r_deg)
            print 'face', (x,y,w,h), center_org

            resized = face_cand
            if w < IMAGE_SIZE or w > IMAGE_SIZE * 2:
                print 'resizing..'
                resized = cv2.resize(face_cand, (IMAGE_SIZE, IMAGE_SIZE))

            result = {
                    'face_id': 'f%s' % face_id_seed,
                    'img_resized': resized, #顔候補bitmap(小さい場合zoom)
                    'img': face_cand, #顔候補bitmap(元サイズ)
                    'deg': deg, #回転
                    'frame': (x, y, w, h), #回転状態における中心座標+size
                    'center_org': center_org, #角度0時における中心座標
                    }
            results.append(result)
            face_id_seed += 1

            #"""
            #out_file = '%s/face_%s_%s.jpg' % (img_dir, result['face_id'], deg, center_org)
            #cv2.imwrite(out_file, face_cand)
            #"""

    eyes_id_seed = 0
    eyes_faces = []

    for result in results:
        print '#eyes:',result['face_id']

        img = np.copy(result["img_resized"])
        fw,fh = img.shape
        eyes = eye_cascade.detectMultiScale(img, 1.02)
        left_eye = right_eye = None #左上/右上にそれぞれ目が１つずつ検出できればOK
        for (x,y,w,h) in eyes:
            print '## eye:',x,y,w,h,fw/8,fw/4,fh/8,fh/4
            cv2.rectangle(img,(x,y),(x+w,y+h),(64,64,0),1)

            if not (fw/6 < w and w < fw/2):
                print 'eye width invalid'
                continue
            if not (fh/6 < h and h < fh/2):
                print 'eye height invalid'
                continue
            if not fh * 0.5 - (y + h * 0.5) > 0: #上半分に存在する
                print 'eye position invalid'
                continue
            if fw * 0.5 - (x + w * 0.5) > 0:
                if left_eye:
                    print 'too many left eye'
                    continue
                else:
                    left_eye = (x,y,w,h)
            else:
                if right_eye:
                    print 'too many right eye'
                    continue
                else:
                    right_eye = (x,y,w,h)
        #"""
        out_file = '%s/eyes_%s_%s_%s.jpg' % (img_dir, result['face_id'], result['deg'], result['center_org'])
        cv2.imwrite(out_file, img)
        #"""
        if left_eye and right_eye:
            print '>>> valid eyes detect'
            result['left_eye'] = left_eye
            result['right_eye'] = right_eye
            eyes_faces.append(result)

    #重複検出を除去
    candidates = []
    for i, result in enumerate(eyes_faces):
        print 'result:',result['face_id']
        result['duplicated'] = False
        for cand in candidates:
            c_x, c_y = cand['center_org']
            _,_,cw,ch = cand['frame']
            r_x, r_y = result['center_org']
            _,_,rw,rh = result['frame']
            if abs(c_x - r_x) < ((cw+rw)*0.5*0.3) and abs(c_y - r_y) < ((ch+rh)*0.5*0.3): #近い場所にある顔候補
                c_diff = eyes_vertical_diff(cand)
                r_diff = eyes_vertical_diff(result)
                print 'c_diff:',cand['face_id'],c_diff
                print 'r_diff:',result['face_id'],r_diff
                if c_diff < r_diff: #より左右の目の水平位置が近いほうが採用
                    result['duplicated'] = True
                else:
                    cand['duplicated'] = True
        candidates.append(result)
    filtered = filter(lambda n: n['duplicated'] == False, candidates)

    res = []
    #"""
    for cand in filtered:
        out_file = '%s/cand_%s_face_%s_%s.jpg' % (img_dir, cand['face_id'], cand['deg'], cand['center_org'])
        cv2.imwrite(out_file, cand['img'])
        #res.append(out_file)
    #"""

    #候補に対してさらに口検出チェック
    for item in filtered:
        img = np.copy(item["img_resized"])
        fw,fh = img.shape
        mouthes = mouth_cascade.detectMultiScale(img, 1.02) #faceの中心下部付近にあればOK
        mouth_found = False
        for (mx,my,mw,mh) in mouthes:
            print 'mouth',(mx,my,mw,mh)
            cv2.rectangle(img,(mx,my),(mx+mw,my+mh),(128,128,0),2)
            h_diff = fh/2 - (my+mh/2)
            print fh, h_diff
            if h_diff < 0:
                mouth_found = True
                break
        #"""
        out_file = '%s/mouth_%s_face_%s_%s.jpg' % (img_dir, item['face_id'], item['deg'], item['center_org'])
        cv2.imwrite(out_file, img)
        #"""
        print item['face_id'], 'mouth found?', mouth_found
        if mouth_found:
            out_file1 = '%s/final_%s_%s_%s_%s.jpg' % (out_dir, fn, item['face_id'], item['deg'], item['center_org'])
            out_file2 = '%s/final_%s_%s_%s_%s.jpg' % (img_dir, fn, item['face_id'], item['deg'], item['center_org'])
            cv2.imwrite(out_file1, item['img'])
            cv2.imwrite(out_file2, item['img'])
            res.append(out_file1)

    return res

def eyes_vertical_diff(face):
    _,ly,_,lh = face["left_eye"]
    _,ry,_,rh = face["right_eye"]
    return abs((ly + lh * 0.5) - (ry + rh * 0.5))

def rotate_coord(pos, origin, deg):
    """
    posをdeg度回転させた座標を返す
    pos: 対象となる座標tuple(x,y)
    origin: 原点座標tuple(x,y)
    deg: 回転角度
    @return: 回転後の座標tuple(x,y)
    @see: http://www.geisya.or.jp/~mwm48961/kou2/linear_image3.html
    """
    x, y = pos
    ox, oy = origin
    r = np.radians(deg)
    xd = ((x - ox) * np.cos(r) - (y - oy) * np.sin(r)) + ox
    xy = ((x - ox) * np.sin(r) + (y - oy) * np.cos(r)) + oy
    return (int(xd), int(xy))

def mouth_nose_check(item, img_dir):
    img = np.copy(item["img_resized"])
    x,y,w,h = item['frame']
    scale = 1.02
    #faceの中心下部付近にあればOK
    mouthes = mouth_cascade.detectMultiScale(img, scale)
    mouth_found = False

    for (mx,my,mw,mh) in mouthes:
        cv2.rectangle(img,(mx,my),(mx+mw,my+mh),(128,128,0),2)

        w_diff = abs(w/2 - (mx+mw/2))
        h_diff = h/2 - (my+mh/2)
        if h_diff < 0 and w_diff < w/10:
            mouth_found = True
            break
    #"""
    out_file = '%s/mouth_%s_face_%s_%s.jpg' % (img_dir, item['face_id'], item['deg'], item['center_org'])
    cv2.imwrite(out_file, img)
    #"""
    print item['face_id'], 'mouth found? ', mouth_found

    return mouth_found

def show(img):
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def validate(img_file):
    print img_file
    img = cv2.imread(img_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale = 1.02
    faces = face_cascade.detectMultiScale(gray, scale, 5)

def detect_face(img_file, outdir = 'out', restrict = True):
    if restrict:
        exts = ['.JPG','.JPEG']
        filename = os.path.basename(os.path.normpath(img_file))
        (fn, ext) = os.path.splitext(filename)
        if not ext.upper() in exts:
            print 'jpg only, skip:%s' % img_file
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

        if debug:
            filename = os.path.basename(os.path.normpath(img_file))
            out_file = 'out/%s' % filename
            cv2.imwrite(out_file, img)
            print out_file

        valid = True
        if restrict:
            valid = restrict_check(gray, img, x, y, w, h, scale, debug)

        if not valid:
            continue

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

    return res

def restrict_check(gray, img, x, y, w, h, scale, debug):

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

    print 'eye:%s mouth:%s nose:%s' % (eyes_ok, mouth_ok, nose_ok)
    return eyes_ok and mouth_ok and nose_ok

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
    #detect_face(param[1])
    detect_face_rotate(param[1], 'out_rotated')
    #validate(param[1])

