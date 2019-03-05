# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description:
"""


import json
from datetime import datetime

import pymysql
import requests
from flask import Flask, redirect, request

app = Flask(__name__)



class ApiError(Exception):
    """ 错误异常处理 """

    def __init__(self, code, msg):
        super(ApiError, self).__init__()
        # 错误状态码
        self.code = code
        # 错误信息
        self.msg = msg

    def __str__(self):
        return '{0}:{1}'.format(self.code, self.msg)


class ServerError(Exception):
    pass

class WeiboClient(object):
    API_URL = 'https://api.weibo.com/'  # 微博api地址

    def __init__(self, client_id, client_secret):
        self.client_id = client_id  # 应用id
        self.client_secret = client_secret  # 应用秘钥
        self.token = {}

    @property
    def access_token(self):
        if self.token:
            return self.token['access_token']
        return None

    def fetch(self, method, url, params={}):
        '''
        接口请求的统一封装
        '''
        try:
            if method == 'POST':
                resp = requests.post(url, params)
            else:
                resp = requests.get(url, params)

            if resp.status_code >= 200 and resp.status_code < 300:
                # 接口正常
                rest = resp.json()
                if 'error_code' in rest:
                    raise ApiError(rest['error_code'], rest['error'])
                return rest
            elif resp.status_code >= 400:
                raise ServerError()
        except ApiError as e:
            print('ApiError')
            pass
        except ServerError as e:
            print('ServerError')
        except Exception:
            print('Exception')


    def get_ticket_url(self, redirect_uri=None):
        '''
        获取从浏览器跳转的 url
        用来获取token
        '''
        if redirect_uri is None:
            redirect_uri = 'http://test.baidu.com'
        url = self.API_URL + 'oauth2/authorize?client_id={0}&response_type=code&redirect_uri={1}'.format(
            self.client_id,
            redirect_uri
        )
        # get请求
        return url

    def get_token(self, code):
        '''
        获取token
        '''
        # 如果已经有了，则直接返回
        if self.token:
            return self.token
        url = self.API_URL + 'oauth2/access_token?client_id={0}&client_secret={1}&grant_type=authorization_code&redirect_uri=http://test.baidu.com&code={2}'.format(
            self.client_id,
            self.client_secret,
            code
        )
        resp = self.fetch('POST', url)
        self.token = resp.json()
        return self.token

    def get_user_info(self, access_token, uid):
        '''
        获取用户信息
        '''
        url = self.API_URL + '2/users/show.json'
        # access_token = self.get_token(code)['access_token']
        resp = self.fetch('GET', url, {
            'access_token': access_token,
            'uid': uid
        })
        return resp.json()

    def get_conn(self):
        """ 获取mysql 的连接 """
        try:
            conn = pymysql.connect(
                db='test',
                host='localhost',
                user='root',
                password='xxxxMMMM3333#',
                charset='utf8'
            )
        except:
            pass
        return conn

    def weibo_share(self):
        '''
        分享数据到微博
        '''

        url = self.API_URL + '2/statuses/share.json'
        resp = self.fetch('POST', url, {
            'status': '现在是北京时间： {0} http://test.baidu.com'.format(datetime.now())
        })
        return resp

client_id = ''
client_secret = ''
client = WeiboClient(client_id, client_secret)


# 登录
@app.route('/')
def index():
    code = request.args.get('code', 200)
    # 根据code来获取token
    token = client.get_token(code)
    # 获取用户信息

    user_info = client.get_user_info(token['access_token'], token['uid'])
    third_id = user_info['id']
    nickname = user_info['screen_name']
    headimg = user_info['profile_image_url']

    # 获取数据库的链接
    conn = client.get_conn()
    cursor = conn.cursor()
    sql = "INSERT INTO `user`(`third_id`, `nickname`, `headimg`) VALUES('{third_id}', '{nickname}', '{headimg}')".format(
        third_id=third_id, nickname=nickname, headimg=headimg)
    cursor.execute(sql)
    conn.autocommit(True)
    return json.dumps(user_info)


# 登录回调获取token
@app.route('/weibo')
def weibo():
    ticket = client.get_ticket_url()
    return redirect(ticket)

# 分享
@app.route('/share')
def share():
    rest = client.weibo_share()
    return json.dumps(rest)

if __name__ == '__main__':
    app.run(debug=True, port=8010)
