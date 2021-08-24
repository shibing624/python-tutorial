# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""


def parse(text):
    # 检测当前 input 解析状态
    result = test_state(input)

    if block_state == BLOCK.Block:
        return result

    # 分析标题标记 #
    title_rank = 0
    for i in range(6, 0, -1):
        if input[:i] == '#' * i:
            title_rank = i
            break
    if title_rank != 0:
        # 处理标题，转化为相应的 HTML 文本
        result = handleTitle(input, title_rank)
        return result

    # 分析分割线标记 --
    if len(input) > 2 and all_same(input[:-1], '-') and input[-1] == '\n':
        result = "<hr>"
        return result

    # 解析无序列表
    unorderd = ['+', '-']
    if result != "" and result[0] in unorderd:
        result = handleUnorderd(result)
        is_normal = False

    f = input[0]
    count = 0
    sys_q = False
    while f == '>':
        count += 1
        f = input[count]
        sys_q = True
    if sys_q:
        result = "<blockquote style=\"color:#8fbc8f\"> " * count + "<b>" + input[
                                                                           count:] + "</b>" + "</blockquote>" * count
        is_normal = False

    # 处理特殊标记，比如 ***, ~~~
    result = tokenHandler(result)
    # END

    # 解析图像链接
    result = link_image(result)
    pa = re.compile(r'^(\s)*$')
    a = pa.match(input)
    if input[-1] == "\n" and is_normal == True and not a:
        result += "</br>"

    return result


with open('c.md', 'r', encoding='utf-8') as f:
    # 逐行解析 markdwon 文件
    for eachline in f:
        result = parse(eachline.strip())
        if result:
            print(result)
