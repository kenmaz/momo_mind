# vim:fileencoding=utf-8

import cv2
import sys
import os

def resize(src, dst):
    print('resize and save',src, dst)
    image = cv2.imread(src)
    image = cv2.resize(image, (112, 112))
    cv2.imwrite(dst, image)

def get_jpg_paths(dir):
    res = []
    exts = ['.JPG','.JPEG']
    for item in os.listdir(dir):
        path = os.path.join(dir, item)
        if os.path.isfile(path):
            fn, ext = os.path.splitext(path)
            if ext.upper() in exts:
                res.append(path)
    return res

def execute(src_dir, dst_dir):
    src_paths = get_jpg_paths(src_dir)
    for src_path in src_paths:
        basename = os.path.basename(src_path)
        fn, ext = os.path.splitext(basename)
        fn = fn.translate(None, '.')
        fn = '%s.jpg' % fn
        dst_path = os.path.join(dst_dir, fn)
        resize(src_path, dst_path)

if __name__ == "__main__":
    param = sys.argv
    execute(param[1], param[2])
