#!/usr/bin/env python
# -*- coding: utf-8 -*-

def application(evn, start_response):
    start_response('200 OK', [('Content-Type','text/html')])
    return "Hello World"

