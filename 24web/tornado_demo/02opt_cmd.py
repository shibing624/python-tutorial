# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import tornado
import tornado.ioloop
import tornado.httpserver
import tornado.options
import tornado.web
tornado.options.define("port", default=9998, type=int, help="server port")
tornado.options.define("names", default=['lili', 'lucy'], type=str, multiple=True, help="names")

class IndexHandle(tornado.web.RequestHandler):
    def get(self, *args, **kwargs):
        self.write("hello names")

if __name__ == '__main__':
    tornado.options.parse_command_line()
    print(tornado.options.options.names)
    app = tornado.web.Application([
        (r"/", IndexHandle),
    ])
    server = tornado.httpserver.HTTPServer(app)
    server.listen(tornado.options.options.port)
    tornado.ioloop.IOLoop.current().start()