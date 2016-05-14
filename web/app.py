#!/usr/bin/env python

import json
from flask import Flask
from flask import render_template
from flask import request, jsonify
from werkzeug import secure_filename
import uuid
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../face_detect')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../deeplearning')
import detect
import mcz_eval

app = Flask(__name__)
app.debug = True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    f = request.files['file']
    filename = secure_filename(f.filename)
    (fn, ext) = os.path.splitext(filename)
    input_path = '/tmp/' + uuid.uuid1().hex + ext
    print input_path
    f.save(input_path)
    faces = detect.detect_face(input_path, '/tmp')
    print faces

    allres = []
    for faceimg in faces:
        res = mcz_eval.main('../deeplearning/data/model.ckpt-15000', faceimg)
        allres.append(res)

    return jsonify({'res':allres})

if __name__ == '__main__':
    app.run()
