# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
from django.http import HttpResponse
from django.shortcuts import render


def hello(request):
    context = {}
    context["hello"] = 'helloo1 world'
    context["say"] = 'say world'
    persons = [{'name': 'lILli', 'score': 30}, {'name': 'lucy', 'score': 50}, {'name': 'tom', 'score': 80}, ]
    context['persons'] = persons
    return render(request, 'hello.html', context)


def index(request):
    return HttpResponse("Hello world ! ")
