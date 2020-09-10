# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import tornado

print(tornado.version_info)

import tornado.ioloop
import tornado.web
import tornado.httpserver

class HandleDemo(tornado.web.RequestHandler):
    def get(self, *args, **kwargs):
        self.write("hello")


class CalcMin(tornado.web.RequestHandler):
    def get(self, *args, **kwargs):
        a = 123
        b = 321
        c = a * b
        self.write("%s * %s = %s" % (a, b, c))


def make_app():
    return tornado.web.Application([
        (r"/", HandleDemo),
        (r"/calc", CalcMin),
    ])


if __name__ == '__main__':
    app = make_app()
    app.listen(9999)

    # 不建议这个多进程，原因是绑定在一个端口，无法有效监控。
    # http_server = tornado.httpserver.HTTPServer(app)
    # http_server.bind(9999)
    # http_server.start(0) # cpu processor

    tornado.ioloop.IOLoop.current().start()
