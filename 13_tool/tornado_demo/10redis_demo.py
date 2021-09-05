# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import json

import redis
import tornado.ioloop
import tornado.web


class FactorialService(object):
    def __init__(self):
        self.cache = redis.StrictRedis("localhost", 6379)  # redis 缓存
        self.key = "factorials"

    def calc(self, n):
        s = self.cache.hget(self.key, str(n))
        if s:
            return int(s), True

        s = 1
        for i in range(1, n):
            s *= i
        self.cache.hset(self.key, str(n), str(s))  # save result to redis
        return s, False


class IndexHandler(tornado.web.RequestHandler):
    def get(self, *args, **kwargs):
        self.write("hello world")


class FactorialHandler(tornado.web.RequestHandler):
    service = FactorialService()

    def get(self, *args, **kwargs):
        n = int(self.get_argument("n") or 1)
        fact, cached = self.service.calc(n)
        result = {
            "n": n,
            "fact": fact,
            "cached": cached
        }
        self.set_header("Content-Type", "application/json; charset=UTF-8")
        self.write(json.dumps(result, ensure_ascii=False))


def make_app():
    return tornado.web.Application([
        (r"/", IndexHandler),
        (r"/fact", FactorialHandler),
    ])


if __name__ == '__main__':
    app = make_app()
    app.listen(8888)
    tornado.ioloop.IOLoop.current().start()
