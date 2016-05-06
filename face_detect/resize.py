# vim:fileencoding=utf-8

import cv2
import sys
import os

def resize(srcdir, outdir):
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    if os.path.exists(srcdir) and os.path.isdir(srcdir):
        for dirpath, dirnames, filenames in os.walk(srcdir):
            for filename in filenames:
                if not filename.startswith('.') and not dirpath.endswith('bulk'):
                    input_path = os.path.join(dirpath, filename)
                    out_path = os.path.join(outdir, filename)
                    print input_path, out_path
                    image = cv2.imread(input_path)
                    image = cv2.resize(image, (112, 112))
                    cv2.imwrite(out_path, image)

if __name__ == "__main__":
    param = sys.argv
    resize(param[1], param[2])
