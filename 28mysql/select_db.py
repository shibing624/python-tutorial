# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

import pymysql


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

    def exe(self):
        self.check_init()
        # 使用cursor()方法获取操作游标
        cursor = self.db.cursor()
        # SQL语句
        sql = "SELECT id, name from test"

        count = 0
        try:
            # 执行SQL语句
            cursor.execute(sql)
            # 获取所有记录列表
            results = cursor.fetchall()
            for row in results:
                id = row[0]
                name = row[1]
                # 打印结果
                print("id=%s,name=%s" % (id, name))
                count += 1
        except Exception as e:
            print("Error: unable to fetch data", e)
        print("[Info]已查询到%d条数据" % count)
        # 关闭数据库连接
        self.db.close()


if __name__ == "__main__":
    my_db = DB("180.76.120.65", "root", "", "mm_images", port=3306)
    my_db.exe()
