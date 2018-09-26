# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

from flask import Flask, render_template, request

# Initialize the Flask application
app = Flask(__name__)


# Default route, print user's IP
@app.route('/')
def index():
    ip = request.remote_addr
    return render_template('ip.html', user_ip=ip)


if __name__ == '__main__':
    app.run()
