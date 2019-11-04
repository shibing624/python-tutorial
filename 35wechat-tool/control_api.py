# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

import os

import itchat

HELP_MSG = "test with music api"
''' add a friend or accept a friend
    for options
        - userName: 'UserName' for friend's info dict
        - status:
            - for adding status should be 2
            - for accepting status should be 3
        - ticket: greeting message
        - userInfo: friend's other info for adding into local storage
    it is defined in components/contact.py
'''

itchat.auto_login(True)
itchat.send(HELP_MSG, 'filehelper')

s = itchat.add_friend('XLBY81111')
print(s)
