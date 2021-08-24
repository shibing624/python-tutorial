# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""


from flask import Flask, request
from flask_restful import Resource,Api
app = Flask(__name__)
api = Api(app)


class Sample(Resource):
    def get(self):
        return {"hello":'world'}

api.add_resource(Sample, '/')

app.run(debug=True)
