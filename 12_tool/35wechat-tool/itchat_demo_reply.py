# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

import itchat

@itchat.msg_register(itchat.content.TEXT)
def text_reply(msg):
    return msg.text

itchat.auto_login(hotReload=True)

itchat.send('Hello, filehelper....4', toUserName='filehelper')

author = itchat.search_friends(name='snail12270')
print(author)
# author.send('greeting, to 弘二! content: 嘻嘻嘻3')

# s = itchat.search_friends()
s = itchat.search_friends(name='弘')
print(s)

s = itchat.search_friends(name='张')
print(s)