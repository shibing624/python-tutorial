# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

from flask_sqlalchemy import SQLAlchemy
from flask import Flask


app = Flask(__name__)
# 此处是配置SQLALCHEMY_DATABASE_URI, 前面的mysql+mysqlconnetor指的是数据库的类型以及驱动类型
# 后面的username,pwd,addr,port,dbname分别代表用户名、密码、地址、端口以及库名
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@127.0.0.1:3306/demo'
# 创建1个SQLAlichemy实例
db = SQLAlchemy(app)


# 定义1个类(由db.Model继承)，注意这个类是数据库真实存在的，因为我是针对已有数据库做转化
# 我的数据库结构见下图 其中role是数据库的一张表名
class role(db.Model):
    # id是主键db.Column是字段名， db.INT是数据类型
    id = db.Column(db.INT, primary_key=True)
    name = db.Column(db.String(99), unique=False)
    name_cn = db.Column(db.String(99), unique=False)

    def __init__(self, id, name, name_cn):
        self.id = id
        self.name = name
        self.name_cn = name_cn

    def __repr__(self):
        return '<User %r>' % self.name

# 初始化role 并插入数据库
test_role1 = role(10, 'supervisol', '超超超超级管理员')
test_role2 = role(11, 'your try', '你试试哦')
db.session.add(test_role1)
db.session.add(test_role2)
db.session.commit()

#查询数据库
a = db.session.query(role).filter_by(id=2).first()  # 查询role表中id为2的第一个匹配项目，用".字段名"获取字段值
print(a)
b = db.session.query(role).all()  # 得到一个list，返回role表里的所有role实例
print(b)
db.session.query(role).filter(role.id == 2).first() # 结果与第一种一致
# 获取指定字段，返回一个生成器 通过遍历来完成相关操作, 也可以强转为list
d = db.session.query(role).filter_by(id=2).values('id', 'name', 'name_cn')
print(list(d))
# 模糊查询
e = db.session.query(role).filter(role.name_cn.endswith('管理员')).all()  # 获取role表中name_cn字段以管理员结尾的所有内容
print(e)
# 修改数据库内容
user = db.session.query(role).filter_by(id=7).first()  # 将role表中id为6的name改为change
user.name = 'change'
db.session.commit()