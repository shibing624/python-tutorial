# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

import os
import jieba

from datetime import datetime
from PIL import Image


# 字节bytes转化kb\m\g
def formatSize(bytes):
    try:
        bytes = float(bytes)
        kb = bytes / 1024
    except:
        print("传入的字节格式不对")
        return "Error"

    if kb >= 1024:
        M = kb / 1024
        if M >= 1024:
            G = M / 1024
            return "%fG" % (G)
        else:
            return "%fM" % (M)
    else:
        return "%fkb" % (kb)


# 获取文件大小
def getDocSize(path):
    try:
        size = os.path.getsize(path)
        return formatSize(size)
    except Exception as err:
        print(err)


def list_all_files(rootdir):
    import os
    _files = []
    list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
    for i in range(0, len(list)):
        path = os.path.join(rootdir, list[i])
        if os.path.isdir(path):
            _files.extend(list_all_files(path))
        if os.path.isfile(path):
            _, basename = os.path.split(path)
            if basename.startswith('.'): continue
            _files.append(path)
    return _files


def get_info(file_paths, count=1):
    info = []
    for i in file_paths:
        relative_path = i
        absolute_path = os.path.abspath(relative_path)
        names = absolute_path.split("/")
        name = names[-1]
        title = names[-2]
        category = names[-3]
        _, suffix = os.path.splitext(absolute_path)
        image_size = "(0,0)"
        try:
            img = Image.open(absolute_path)
            # print(img.size)
            image_size = str(img.size)
        except Exception as e:
            print('image open error', e)
            pass
        storage_size = getDocSize(absolute_path)
        line_list = [count, category, title, absolute_path, relative_path, suffix, "timo", str(image_size),
                     str(storage_size), datetime.now(), name, 0, 0, 0, ' '.join(jieba.lcut(title))]
        info.append(line_list)
        count += 1

    return info


if __name__ == "__main__":
    path = './data'
    file_paths = list_all_files(path)
    info = get_info(file_paths)
    print(info)
