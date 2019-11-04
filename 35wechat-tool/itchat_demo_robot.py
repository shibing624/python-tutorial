# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

import time

import itchat
import requests

KEY = '7c8cdb56b0dc4450a8deef30a496bd4c'  # 图灵机器人key值,这里也可以是其他供应商机器人的key值,比如微软小冰.


def get_response(msg):
    # 构造了要发送给图灵服务器的数据
    apiUrl = 'http://www.tuling123.com/openapi/api'  # 图灵API
    data = {
        'key': KEY,
        'info': msg,
        'userid': 'wechat-robot',
    }
    # 字典出现异常的情况下会抛出异常,为了防止中断程序,这里使用了try-except异常模块
    try:
        r = requests.post(apiUrl, data=data).json()
        return r.get('text')
    except:
        # 将会返回一个None
        return


# 这里是我们在实现微信消息的获取的同时直接回复
@itchat.msg_register(itchat.content.TEXT)
def tuling_reply(msg):
    time.sleep(3)  # 延时函数
    # 为了保证在图灵Key出现问题的时候仍旧可以回复，这里设置一个默认回复
    defaultReply = 'I received: ' + msg['Text']
    # 如果图灵Key出现问题，那么reply将会是None
    reply = get_response(msg['Text'])
    return reply or defaultReply


# 为了不重复扫码登陆,这里使用热启动
itchat.auto_login(hotReload=True)
friend = itchat.search_friends("snail12270")[0]
friend_username = friend['UserName']
itchat.send('你可以提现吗？', toUserName=friend_username)

print(friend['UserName'])
itchat.set_alias(friend_username, 'lili01')

f = itchat.search_friends('lili01')
print(f)
itchat.run()
