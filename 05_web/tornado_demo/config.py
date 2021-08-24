# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os

pwd_path = os.path.abspath(os.path.dirname(__file__))
print(os.path.join(pwd_path, 'templates'))
settings = {
    'template_path': os.path.join(pwd_path, 'templates'),
    'static_path': os.path.join(pwd_path, 'statics'),
    'cookie_secret': 'abc123_Oopb',  #混淆加密
    'xsrf_cookies': True, # 开启xsrf保护
    'login_url': '/login',
    'debug': True,
}
log_path = os.path.join(pwd_path, 'tornado.log')
options = {
    "port":8888,

}
