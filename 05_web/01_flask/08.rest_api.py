# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

from flask import request
from flask import Flask, jsonify
from flask import abort

from flask import make_response
from flask import url_for

from flask_httpauth import HTTPBasicAuth
from flask_restful import Api, Resource

auth = HTTPBasicAuth()
app = Flask(__name__)

api = Api(app)

class HelloWorld(Resource):
    def get(self):
        return {'Hello':'world'}

api.add_resource(HelloWorld,'/')



tasks = [
    {
        'id': 1,
        'title': u'Buy groceries',
        'description': u'Milk, Cheese, Pizza, Fruit, Tylenol',
        'done': False
    },
    {
        'id': 2,
        'title': u'Learn Python',
        'description': u'Need to find a good Python tutorial on the web',
        'done': False
    }
]

class TodoSimple(Resource):
    # 获取资源
    def get(self,todo_id):
        return {todo_id:tasks[todo_id]}

    # 更新资源
    def put(self,todo_id):
        tasks[todo_id] = request.form['data']
        return {todo_id:tasks[todo_id]}

api.add_resource(TodoSimple,'/<int:todo_id>')

def make_public_task(task):
    new_task = {}
    for field in task:
        if field == 'id':
            new_task['uri'] = url_for('get_task', task_id=task['id'], _external=True)
        else:
            new_task[field] = task[field]
    return new_task


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


@auth.get_password
def get_password(username):
    if username == 'ok':
        return 'python'
    return None


@auth.error_handler
def unauthorized():
    return make_response(jsonify({'error': 'Unauthorized access'}), 401)


@app.route('/todo/api/v1.0/tasks/<int:task_id>', methods=['GET'])
def get_task(task_id):
    task = None
    for i in tasks:
        if i['id'] == task_id:
            task = i
    if not task:
        not_found(404)
    return jsonify({'task': task})


@app.route('/todo/api/v1.0/tasks', methods=['POST'])
def create_task():
    if not request.json or not 'title' in request.json:
        not_found(400)
    task = {
        'id': tasks[-1]['id'] + 1,
        'title': request.json['title'],
        'description': request.json.get('description', ""),
        'done': False
    }
    tasks.append(task)
    return jsonify({'task': task}), 201


@app.route('/todo/api/v1.0/tasks', methods=['GET'])
@auth.login_required
def get_tasks():
    return jsonify({'tasks': list(map(make_public_task, tasks))})


@app.route('/todo/api/v1.0/tasks/<int:task_id>', methods=['PUT'])
def update_task(task_id):
    task = None
    for i in tasks:
        if i['id'] == task_id:
            task = i
    if not task:
        abort(404)
    # if not request.json:
    #     abort(400)
    # if 'title' in request.json:
    #     abort(400)
    # if 'description' in request.json:
    #     abort(400)
    # if 'done' in request.json and type(request.json['done']) is not bool:
    #     abort(400)
    # task[0]['title'] = request.json.get('title', task[0]['title'])
    # task[0]['description'] = request.json.get('description', task[0]['description'])
    # task[0]['done'] = request.json.get('done', task[0]['done'])
    # return jsonify({'task': task[0]})
    task["title"] = "new one"
    return jsonify({'task': task})


@app.route('/todo/api/v1.0/tasks_new/<int:task_id>', methods=['PUT'])
def update_task_new(task_id):
    task = filter(lambda t: t['id'] == task_id, tasks)
    task = list(task)
    if len(task) == 0:
        abort(404)
    task[0]['title'] = 'new two'
    return jsonify({'task': task[0]})


@app.route('/todo/api/v1.0/tasks/<int:task_id>', methods=['DELETE'])
def delete_task(task_id):
    task = filter(lambda t: t['id'] == task_id, tasks)
    task = list(task)
    if len(task) == 0:
        abort(404)
    tasks.remove(task[0])
    return jsonify({'result': True})


if __name__ == '__main__':
    app.run(debug=True)
