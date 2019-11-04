# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

from markdown import markdown,markdownFromFile


def gen(md_text):
    r = markdown(md_text)
    print(r)
    return r

def parse_file(input_path,output_path):
    r = markdownFromFile(input=input_path, output=output_path,encoding='utf-8')

parse_file('b.md','c.html')
