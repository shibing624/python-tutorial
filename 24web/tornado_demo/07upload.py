# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import tornado.web
import tornado.ioloop
import tornado.httpserver
import tornado.options
from tornado.options import options, define
from tornado.web import RequestHandler, MissingArgumentError
from tornado.web import url
define("port", default=8888, type=int, help="run server on the given port.")


class IndexHandler(RequestHandler):
    def get(self, *args, **kwargs):
        self.write("hello")


class UploadHandler(RequestHandler):
    def post(self, *args, **kwargs):
        files = self.request.files
        img_files = files.get('img')
        if img_files:
            img_file = img_files[0]['body']
            with open('./itcast.txt', 'w+') as f:
                f.write(str(img_file) + "\n")
            self.write("Ok")

if __name__ == "__main__":
    tornado.options.parse_command_line()
    app = tornado.web.Application([
        (r"/", IndexHandler),
        (r"/upload", UploadHandler),
        url(r"/app", tornado.web.RedirectHandler,
                    dict(url="http://www.baidu.com")),
    ])
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.current().start()
