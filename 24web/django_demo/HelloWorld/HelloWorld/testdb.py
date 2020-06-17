# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

# -*- coding: utf-8 -*-

from django.http import HttpResponse

from TestModel.models import RiskData


# 数据库操作
def testdb(request):
    test1 = RiskData(title='runoob')
    test1.save()
    return HttpResponse("<p>数据添加成功！</p>")


# 数据库操作
def show_db(request):
    # 初始化
    response = ""
    response1 = ""

    # 通过objects这个模型管理器的all()获得所有数据行，相当于SQL中的SELECT * FROM
    list = RiskData.objects.all()

    # filter相当于SQL中的WHERE，可设置条件过滤结果
    response2 = RiskData.objects.filter(id=1)

    # 获取单个对象
    response3 = RiskData.objects.get(id=1)

    # 限制返回的数据 相当于 SQL 中的 OFFSET 0 LIMIT 2;
    RiskData.objects.order_by('userid')[0:2]

    # 数据排序
    RiskData.objects.order_by("id")

    # 上面的方法可以连锁使用
    RiskData.objects.filter(title="runoob").order_by("id")

    # 输出所有数据
    for var in list:
        response1 += var.title + " " + var.show_url
    response = response1
    return HttpResponse("<p>" + response + "</p>")


# 数据库操作
def update_db(request):
    # 修改其中一个id=1的name字段，再save，相当于SQL中的UPDATE
    test1 = RiskData.objects.get(id=1)
    test1.name = 'Google'
    test1.save()

    # 另外一种方式
    # Test.objects.filter(id=1).update(name='Google')

    # 修改所有的列
    # Test.objects.all().update(name='Google')

    return HttpResponse("<p>修改成功</p>")