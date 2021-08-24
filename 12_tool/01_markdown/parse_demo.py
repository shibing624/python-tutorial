# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

import mistletoe

with open('c.md','r') as f:
    rendered = mistletoe.markdown(f)
    print(rendered)

from mistletoe.latex_renderer import LaTeXRenderer
with open('c.md','r') as f:
    print()
    r = mistletoe.markdown(f,LaTeXRenderer)
    print(r)

# coding: utf-8

import os
import time

NUM = 100


def benchmark_misaka(text):
    import misaka as m
    # mistune has all these features
    extensions = (
        m.EXT_NO_INTRA_EMPHASIS | m.EXT_FENCED_CODE | m.EXT_AUTOLINK |
        m.EXT_TABLES | m.EXT_STRIKETHROUGH
    )
    md = m.Markdown(m.HtmlRenderer(), extensions=extensions)
    t0 = time.time()
    for i in range(NUM):
        md.renderer(text)
    t1 = time.time()
    print('misaka', (t1 - t0) * 1000, 'ms')


def benchmark_markdown2(text):
    import markdown2
    extras = ['code-friendly', 'fenced-code-blocks', 'footnotes']
    t0 = time.time()
    r=''
    for i in range(NUM):
        r = markdown2.markdown(text, extras=extras)
    print(r)
    t1 = time.time()
    print('markdown2', (t1 - t0), 's')


def benchmark_markdown(text):
    import markdown
    t0 = time.time()
    for i in range(NUM):
        markdown.markdown(text, ['extra'])
    t1 = time.time()
    print('markdown', (t1 - t0), 's')


if __name__ == '__main__':
    root = os.path.dirname(__file__)
    filepath = os.path.join(
        root, 'c.md'
    )
    print(filepath)
    with open(filepath, 'r') as f:
        text = f.read()
        print(text)

    # benchmark_misaka(text)
    benchmark_markdown2(text)
    benchmark_markdown(text)