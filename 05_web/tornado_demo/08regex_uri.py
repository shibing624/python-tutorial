# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import tornado.ioloop
import tornado.web
import tornado.options
from tornado.options import options, define
from tornado.web import RequestHandler

define("port", default=8000, type=int, help="run server on the given port.")


class IndexHandler(RequestHandler):
    def get(self, *args, **kwargs):
        self.write("hello")


class SubjectCityHandler(RequestHandler):
    def get(self, subject, city, *args, **kwargs):
        self.write(("subject: %s<br/>City:%s" % (subject, city)))


class SubjectDateHandler(RequestHandler):
    def get(self, date, subject, *args, **kwargs):
        self.write(("Date:%s <br/>Subject: %s" % (date, subject)))


if __name__ == '__main__':
    tornado.options.parse_command_line()
    app = tornado.web.Application([
        (r"/", IndexHandler),
        (r"/sub-city/(.+)/([a-z]+)", SubjectCityHandler),
        (r"/sub-date/(?P<subject>.+)/(?P<date>\d+)", SubjectDateHandler), #　命名方式, 更好
    ])
    app.listen(options.port)
    tornado.ioloop.IOLoop.current().start()
