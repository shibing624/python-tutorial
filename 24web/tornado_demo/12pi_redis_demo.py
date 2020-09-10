# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import json
import math
import redis
import tornado.ioloop
import tornado.web


class FactorialService(object):
    def __init__(self, cache):
        self.cache = cache
        self.key = "factorials"

    def calc(self, n):
        s_dict = self.cache.get(self.key, {})
        if s_dict:
            s = s_dict.get(n, None)
            if s:
                return s, True
        s = 1
        for i in range(1, n):
            s *= i

        self.cache.setdefault(self.key, {})
        self.cache[self.key].setdefault(n, s)
        return s, False


class PiService(object):
    def __init__(self, cache):
        self.cache = cache
        self.key = "pis"

    def calc(self, n):
        s_dict = self.cache.get(self.key, {})
        if s_dict:
            s = s_dict.get(n, None)
            if s:
                return s, True
        s = 0.0
        for i in range(n):
            s += 1.0 / (2 * i + 1) / (2 * i + 1)
        s = math.sqrt(s * 8)
        self.cache.setdefault(self.key, {})
        self.cache[self.key].setdefault(n, s)
        return s, False


class IndexHandler(tornado.web.RequestHandler):
    def get(self, *args, **kwargs):
        self.write("hello world")


class FactorialHandler(tornado.web.RequestHandler):
    def initialize(self, model):
        self.model = model

    def get(self, *args, **kwargs):
        n = int(self.get_argument("n") or 1)
        fact, cached = self.model.calc(n)
        result = {
            "n": n,
            "fact": fact,
            "cached": cached
        }
        self.set_header("Content-Type", "application/json; charset=UTF-8")
        self.write(json.dumps(result, ensure_ascii=False))


class PiHandler(tornado.web.RequestHandler):
    def initialize(self, model):
        self.model = model

    def get(self, *args, **kwargs):
        n = int(self.get_argument("n") or 1)
        pi, cached = self.model.calc(n)
        result = {
            "n": n,
            "pi": pi,
            "cached": cached
        }
        self.set_header("Content-Type", "application/json; charset=UTF-8")
        self.write(json.dumps(result, ensure_ascii=False))


def make_app():
    cache = dict()
    factorial_service = FactorialService(cache)
    pi_service = PiService(cache)

    return tornado.web.Application([
        (r"/", IndexHandler),
        (r"/fact", FactorialHandler, {"model": factorial_service}),
        (r"/pi", PiHandler, {"model": pi_service}),
    ])


if __name__ == '__main__':
    app = make_app()
    app.listen(8888)
    tornado.ioloop.IOLoop.current().start()
