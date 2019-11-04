# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

import mistune
markdown = mistune.Markdown()
a = markdown('I am using **mistune markdown parser**')
print(a)
print(mistune.markdown('I am using **mistune markdown parser**'))

renderer = mistune.Renderer(escape=True, hard_wrap=True)
# use this renderer instance
markdown = mistune.Markdown(renderer=renderer)
print(markdown('hi hello'))
with open("a.md",'w',encoding='utf-8')as f:
    f.write(a)

