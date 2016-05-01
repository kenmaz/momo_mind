#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os

if __name__ == '__main__':
  outdir = sys.argv[1]

  if not os.path.isdir(outdir):
    sys.exit('%s is not directory' % outdir)

  names = {
    "reni": 0,
    "kanako": 1,
    "shiori": 2,
    "arin": 3,
    "momoka": 4,
  }

  #exts = ['.PNG','.JPG','.JPEG']
  exts = ['.JPG','.JPEG']

  for dirpath, dirnames, filenames in os.walk(outdir):
    for dirname in dirnames:
      if dirname in names:
        n = names[dirname]
        member_dir = os.path.join(dirpath, dirname)
        for dirpath2, dirnames2, filenames2 in os.walk(member_dir):
          if not dirpath2.endswith(dirname):
            continue
          for filename2 in filenames2:
            (fn,ext) = os.path.splitext(filename2)
            if ext.upper() in exts:
              img_path = os.path.join(dirpath2, filename2)
              print '%s %s' % (img_path, n)

