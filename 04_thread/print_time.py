# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

import time
import threading


def send_online_notification(user):
    print(user)
    while True:
        print('I\'m Still Alive!! ' + time.strftime('%y/%m/%d-%H:%M:%S', time.localtime()))
        time.sleep(5)


username = 'x'
t = threading.Thread(target=send_online_notification, args=(username,))
t.setDaemon(True)
t.start()

# embed()
print('bot embed.')
while True:
    print("a")
    time.sleep(2)
