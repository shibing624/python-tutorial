# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

# __init__.py
from flask import Flask, current_app, make_response, request
from flask_caching import Cache
import random
app = Flask(__name__)
app.config['SECRET_KEY'] = '123'
cache = Cache()
cache.init_app(app, config={'CACHE_TYPE': 'simple'})


@app.route('/test1')
@cache.cached(timeout=5,key_prefix='index')
def test():
    cache.set('name', 'xiaoming', timeout=30)
    cache.set('person', {'name': 'aaa', 'age': 20})
    x = cache.get('name')
    print(x)
    cache.set_many([('name1', 'hhh'), ('name2', 'jjj')])
    print(cache.get_many("name1", "name2"))
    print(cache.delete("name"))
    print(cache.delete_many("name1", "name2"))
    y = cache.get("name")
    print(y)
    z = random.randint(0,100)
    print(z)
    return str(x + " " + str(z))


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8055, debug=True)
