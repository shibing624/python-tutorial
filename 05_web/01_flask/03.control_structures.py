# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

from flask import Flask, render_template

app = Flask(__name__)


@app.route('/')
@app.route('/<user>')
def user(user=None):
    return render_template('user.html', user=user)


@app.route('/shopping')
def shopping():
    food = ['chinese', 'beef']
    return render_template('shopping.html', food=food)


@app.route('/profile/<name>')
def profile(name):
    return render_template('profile.html', name=name)


app.run()
