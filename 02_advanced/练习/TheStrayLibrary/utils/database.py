from tkinter import *
import sqlite3
import os

class database():
    def addmysql(name, author, comment, state):#添加数据
        Desktoppath = './StrayLibrary/book.db'
        db = sqlite3.connect(Desktoppath)# 使用cursor()方法获取操作游标 
        cursor = db.cursor()# SQL 插入语句
        sql = "INSERT INTO EMPLOYEE(name,author,comment,state)VALUES ('{}','{}','{}','{}')".format(name, author, comment, state)
        try:# 执行sql语句
            cursor.execute(sql)# 提交到数据库执行
            db.commit()
        except:# Rollback in case there is any error
            db.rollback()
        db.close()# 关闭数据库连接

    def changemysql(state,name):#更改数据状态
        Desktoppath = './StrayLibrary/book.db'
        db = sqlite3.connect(Desktoppath)
        cursor = db.cursor()# 使用cursor()方法获取操作游标 
        sql = "UPDATE EMPLOYEE SET state = '%s' where name = '%s' "%(state,name)
        try:
            cursor.execute(sql)
            db.commit()
        except:
            pass
        db.close()

    def checkmysql():#检索数据库
        Desktoppath = './StrayLibrary/book.db'
        db = sqlite3.connect(Desktoppath)
        cursor = db.cursor()# 使用cursor()方法获取操作游标 
        sql = "SELECT * FROM EMPLOYEE"  # SQL 查询语句
        try:
            cursor.execute(sql)# 获取所有记录列表
            results = cursor.fetchall()
            return results
        except:
            pass
        db.close()

    def bulildmysql():
        try:
            os.makedirs("./StrayLibrary") #创建一个文件夹
            Desktoppath = './StrayLibrary/book.db'#文件夹下创建一个数据库
            file=open(Desktoppath,'w')
            file.close()

            db = sqlite3.connect(Desktoppath)
            cursor = db.cursor()# 使用cursor()方法获取操作游标 
            cursor.execute("DROP TABLE IF EXISTS EMPLOYEE")# 如果数据表已经存在使用 execute() 方法删除表。
            sql = """CREATE TABLE EMPLOYEE (name  TEXT(255),author  TEXT(255),comment TEXT(255),state TEXT(255))"""
            cursor.execute(sql)# 创建数据表SQL语句
            db.close()
            database.addmysql('惶然录','费尔南多·佩索阿','一个迷失方向且濒于崩溃的灵魂的自我启示、一首对默默无闻、失败、智慧、困难和沉默的赞美诗。','未借出')
            database.addmysql('以箭为翅','简媜','调和空灵文风与禅宗境界，刻画人间之缘起缘灭。像一条柔韧的绳子，情这个字，不知勒痛多少人的心肉。','未借出')
            database.addmysql('心是孤独的猎手','卡森·麦卡勒斯','我们渴望倾诉，却从未倾听。女孩、黑人、哑巴、醉鬼、鳏夫的孤独形态各异，却从未退场。','已借出')
        except:
            pass