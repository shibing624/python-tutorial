# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
import os
import time
from multiprocessing import Process


def pull_screenrecord(a, b):
    time.sleep(10)
    print(a, b)


def add_friend(a, b, c):
    time.sleep(5)
    print(a, b, c)


class SyncProcess(Process):
    def __init__(self, username, userid='', search_pic_path='', pic_path='', video_path='', time_limit=60,
                 is_video=False):
        super().__init__()
        self.username = username
        self.userid = str(userid)
        self.search_pic_path = search_pic_path
        self.pic_path = pic_path
        self.video_path = video_path
        self.time_limit = time_limit
        self.is_video = is_video

    def run(self):
        print(os.getpid())
        if self.is_video:
            pull_screenrecord(self.video_path, self.time_limit)
        else:
            add_friend(self.username, self.search_pic_path, self.pic_path)


if __name__ == '__main__':
    friend_p = SyncProcess('ll', '123', 'a.png', 'out', 'out/a.mp4', is_video=False)
    friend_p.start()
    friend_p.join()
    print('done')
