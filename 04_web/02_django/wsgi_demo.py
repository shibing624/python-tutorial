# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

def application(env, start_response):
    start_response('200 OK', [('Content-Type','text/html')])
    print("OK")
    return "Hello World"