# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

from flask import Flask, request
from flask_restful import Api, Resource

app = Flask(__name__)
# 创建一个 Api 对象，把 app 作为参数
api = Api(app)

# 创建 Welcome 类，描述欢迎信息(框架可以序列化任意类型的对象)
class Welcome:

    def __init__(self, name):
        self.name = name
        self.message = "Hello %s, Welcome to flask-restaction!" % name

# 创建一个 Hello 类，定义 get 方法
class Hello:
    """Hello world"""

    # 在 get 方法文档字符串中描述输入参数和输出的格式
    def get(self, name):
        """
        Get welcome message

        $input:
            name?str&default="world": Your name
        $output:
            message?str: Welcome message
        """
        return Welcome(name)

class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}

api.add_resource(HelloWorld, '/')



todos = {}

class TodoSimple(Resource):
    def get(self, todo_id):
        return {todo_id: todos[todo_id]}

    def put(self, todo_id):
        todos[todo_id] = request.form['data']
        return {todo_id: todos[todo_id]}

api.add_resource(TodoSimple, '/<string:todo_id>')


if __name__ == '__main__':
    app.run(debug=True)