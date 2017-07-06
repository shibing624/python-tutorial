#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: XuMing <shibing624@126.com>
@description:
"""

from http.server import BaseHTTPRequestHandler, HTTPServer
import time


class RequestHandler(BaseHTTPRequestHandler):
    Page = '''<html>
    <head><title>Title is here.</title></head>
    <body>
    <p>Hello, web!</p>
    </body>
    </html>
    
    '''

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html".encode())
        self.send_header("Content-Length", str(len(self.Page)))
        self.end_headers()
        self.wfile.write(self.Page.encode())


if __name__ == '__main__':
    serverAddress = ('localhost', 9000)
    appServer = HTTPServer(serverAddress, RequestHandler)
    print(time.asctime(), "server starts : %s:%s" % (serverAddress))
    try:
        appServer.serve_forever()
    except KeyboardInterrupt:
        pass
    appServer.server_close()
    print(time.asctime(), "server stops : %s:%s" % (serverAddress))
