# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import tornado.web
import tornado
from tornado_demo import config
import tornado.ioloop
import tornado.httpserver


class IndexHandle(tornado.web.RequestHandler):
    def get(self, *args, **kwargs):
        self.write("hello names file")


if __name__ == '__main__':
    app = tornado.web.Application([
        (r"/", IndexHandle),
    ])
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.bind(config.options["port"])
    http_server.start(1)
    tornado.ioloop.IOLoop.current().start()
