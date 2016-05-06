#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import shutil

def copy(outdir, dirname, memberdir, out_filename):
    dst_dir = os.path.join(outdir, dirname, memberdir)
    dst_path = os.path.join(dst_dir, out_filename)
    src_path = os.path.join(outdir, out_filename)

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    if not os.path.exists(src_path):
        return

    print 'move %s %s' % (src_path, dst_path)
    shutil.move(src_path, dst_path)

def execute(dstdir, srcdir, src_dirname, src_memberdir, src_filenames):
    basenames = [os.path.splitext(f)[0]+'jpg' for f in src_filenames]
    for dst_dirpath, dst_dirnames, dst_filenames in os.walk(dstdir):
        for dst_filename in dst_filenames:
            if dst_filename in basenames:
                copy(dstdir, src_dirname, src_memberdir, dst_filename)

def start(dstdir, srcdir):
    for src_dirpath, src_dirnames, src_filenames in os.walk(srcdir):
        for src_dirname in src_dirnames:
            if src_dirname in ['test', 'train']:
                for src_dirpath2, src_dirnames2, src_filenames2 in os.walk(os.path.join(src_dirpath, src_dirname)):
                    for src_memberdir in src_dirnames2:
                        for src_dirpath3, src_dirnames3, src_filenames3 in os.walk(os.path.join(src_dirpath2, src_memberdir)):
                            execute(dstdir, srcdir, src_dirname, src_memberdir, src_filenames3)

if __name__ == '__main__':
    dstdir = sys.argv[1]
    srcdir = sys.argv[2]
    start(dstdir, srcdir)

