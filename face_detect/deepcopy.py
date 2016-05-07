#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import shutil

def move(src_path, dst_dir):
    basename = os.path.basename(src_path)
    fn, ext = os.path.splitext(basename)
    filename = '%s.jpg' % fn
    dst_path = os.path.join(dst_dir, filename)
    print('move',src_path, dst_path)
    if os.path.exists(dst_path):
        print('%s already exists rm src_path %s' % (dst_path, src_path))
        os.remove(src_path)
    else:
        shutil.move(src_path, dst_dir)
        print('moved',src_path, dst_path)

def get_base_filename(path):
    basename = os.path.basename(path)
    fn, ext = os.path.splitext(basename)
    return fn.translate(None, '.')

def get_jpg_files(dir):
    res = []
    exts = ['.JPG','.JPEG']
    for item in os.listdir(dir):
        path = os.path.join(dir, item)
        if os.path.isfile(path):
            fn, ext = os.path.splitext(path)
            if ext.upper() in exts:
                res.append(path)
    return res

def execute(srcdir, refdir, dstdir):
    srcfiles = get_jpg_files(srcdir)
    reffiles = get_jpg_files(refdir)
    for srcpath in srcfiles:
        print 'try..', srcpath
        for refpath in reffiles:
            if get_base_filename(srcpath) == get_base_filename(refpath):
                print 'match...', refpath
                move(srcpath, dstdir)
                break

def main():
    srcdir = sys.argv[1]
    refdir = sys.argv[2]
    dstdir = sys.argv[3]
    execute(srcdir, refdir, dstdir)

"""
srcdir以下のファイルのうち、refdir以下のファイルと同じ名前を持つものを、dstdirに移動する
"""
if __name__ == '__main__':
    main()

