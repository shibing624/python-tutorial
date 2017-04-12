# -*- coding: utf-8 -*-
"""
@description: 
@author:XuMing
"""
from __future__ import print_function  # 兼容python3的print写法
from __future__ import unicode_literals  # 兼容python3的编码处理

# PyMySQL是一个纯Python写的MySQL客户端，它的目标是替代MySQLdb，
# PyMySQL的性能和MySQLdb几乎相当，如果对性能要求不是特别的强，
# 使用PyMySQL将更加方便。
import pymysql

config = {
    'host': '127.0.0.1',
    'port': 3306,
    'user': 'root',
    'passwd': '',
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor
}
conn = pymysql.connect(**config)
conn.autocommit(1)
cursor = conn.cursor()

try:
    # 创建数据库
    DB_NAME = 'test'
    cursor.execute('DROP DATABASE IF EXISTS %s' % DB_NAME)
    cursor.execute('CREATE DATABASE IF NOT EXISTS %s' % DB_NAME)
    conn.select_db(DB_NAME)

    # 创建表
    TABLE_NAME = 'user'
    cursor.execute('CREATE TABLE %s(id int primary key,name varchar(30))' % TABLE_NAME)

    # 批量插入纪录
    values = []
    for i in range(20):
        values.append((i, 'kk' + str(i)))
    cursor.executemany('INSERT INTO user values(%s,%s)', values)

    # 查询数据条目
    count = cursor.execute('SELECT * FROM %s' % TABLE_NAME)
    print('total records:', cursor.rowcount)

    # 获取表名信息
    desc = cursor.description
    print("%s %3s" % (desc[0][0], desc[1][0]))

    cursor.scroll(10, mode='absolute')
    results = cursor.fetchall()
    for result in results:
        print(result)

except:
    import traceback

    traceback.print_exc()
    # 发生错误时会滚
    conn.rollback()
finally:
    # 关闭游标连接
    cursor.close()
    # 关闭数据库连接
    conn.close()
