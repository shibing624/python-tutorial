# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

from flask import Flask, request

app = Flask(__name__)


@app.route('/')
def index():
    return 'Home page'


@app.route('/hello')
def hello():
    return '<H1>app server</H1>'


@app.route('/user/<username>')
def show_user_name(username):
    return 'Hey baby %s' % username


@app.route('/post/<int:id>')
def show_id(id):
    return 'Show ID %d' % id


@app.route('/method')
def method():
    return "Method use: %s" % request.method


@app.route('/choose', methods=['POST', 'GET'])
def choose():
    if request.method == 'POST':
        return 'You use POST'
    else:
        return 'YOU use GET'


if __name__ == "__main__":
    app.run(debug=True)
