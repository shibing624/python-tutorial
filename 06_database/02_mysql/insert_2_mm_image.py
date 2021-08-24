# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
import sys

sys.path.append(".")
import pymysql

from get_image_info import get_info, list_all_files


def normal_str(text):
    return text.replace('\'', '\\\'')


class DB(object):
    def __init__(self, host="", user_name="", user_password="", db_name="", port=3306, charset='utf8'):
        self.host = host
        self.user_name = user_name
        self.user_password = user_password
        self.db_name = db_name
        self.port = port
        self.charset = charset
        self.is_inited = False

    def init(self):
        # 打开数据库连接
        self.db = pymysql.connect(self.host, self.user_name, self.user_password, self.db_name, charset=self.charset,
                                  port=self.port)
        self.is_inited = True

    def check_init(self):
        if not self.is_inited:
            self.init()

    def exe(self, data=[]):
        self.check_init()
        # 使用cursor()方法获取操作游标
        cursor = self.db.cursor()

        count = 0
        for line in data:
            count += 1

            terms = line  # .strip().split("\t")
            # SQL 插入语句
            sql = "INSERT INTO mm_image_info(id,category,title,absolute_path,relative_path,file_suffix,user,size,storage_space, store_time,name,like_num,unlike_num,collection_num,title_segment) VALUES ('%d','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%d','%d','%d','%s')" % \
                  (terms[0], terms[1], terms[2], terms[3], terms[4], terms[5], terms[6], terms[7], terms[8], terms[9],
                   terms[10], terms[11], terms[12], terms[13], terms[14])
            try:
                # 执行sql语句
                cursor.execute(sql)
                # 提交到数据库执行
                self.db.commit()
            except Exception as e:
                print("error=%s, insert error" % e)
                # Rollback in case there is any error
                self.db.rollback()

            if (count % 10000) == 0:
                print("[Info]已处理%d条数据" % count)
        print("[Info]已处理%d条数据" % count)
        # 关闭数据库连接
        self.db.close()


if __name__ == "__main__":
    path = './data'
    file_paths = list_all_files(path)
    info = get_info(file_paths)
    print(info)

    my_db = DB("180.76.120.65", "root", "", "mm_images", port=3306)
    my_db.exe(info)
