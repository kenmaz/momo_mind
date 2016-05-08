#!/usr/bin/env python

from flask import Flask 

app = Flask(__name__)

@app.route('/')
def index():
    return "<span style='color:red'>I am app 1</span>"
