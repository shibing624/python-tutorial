# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

from flask import Flask, render_template

app = Flask(__name__)


@app.route('/profile/<name>')
def profile(name):
    return render_template('profile.html', name=name)


app.run()
