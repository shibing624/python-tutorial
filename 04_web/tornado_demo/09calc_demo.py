# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import tornado.ioloop
import tornado.web


class FactorialService(object):
    def __init__(self):
        self.cache = {}

    def calc(self, n):
        if n in self.cache:
            return self.cache[n]
        s = 1
        for i in range(1, n):
            s *= i
        self.cache[n] = s
        return s


class IndexHandler(tornado.web.RequestHandler):
    def get(self, *args, **kwargs):
        self.write("hello world")


class FactorialHandler(tornado.web.RequestHandler):
    service = FactorialService()

    def get(self, *args, **kwargs):
        n = int(self.get_argument("n"))
        self.write("%s!=%s" % (n, str(self.service.calc(n))))


def make_app():
    return tornado.web.Application([
        (r"/", IndexHandler),
        (r"/fact", FactorialHandler),
    ])


if __name__ == '__main__':
    app = make_app()
    app.listen(9999)
    tornado.ioloop.IOLoop.current().start()
