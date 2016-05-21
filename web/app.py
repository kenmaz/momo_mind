#!/usr/bin/env python

import json
from flask import Flask
from flask import render_template
from flask import request, jsonify
from werkzeug import secure_filename
import uuid
import sys, os

web_dir = os.path.dirname(os.path.abspath(__file__))
face_detect_dir = web_dir + '/../face_detect'
deeplearning_dir = web_dir + '/../deeplearning'
sys.path.append(face_detect_dir)
sys.path.append(deeplearning_dir)
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

    faces = detect.detect_face_rotate(input_path, web_dir + '/static/tmp')
    print faces

    #res = mcz_eval.main(faces, '../deeplearning/data/model.ckpt-15000')
    res = mcz_eval.execute(faces, deeplearning_dir + '/data/model.ckpt-13000_85per_input56_conv3_fc2')

    return jsonify({'results':res})

if __name__ == '__main__':
    app.run()
    #app.run(host='0.0.0.0')
