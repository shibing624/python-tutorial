# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 

from fp_node import treeNode
def create_tree(data_set, min_support=1):
    """
    创建FP树
    :param data_set: 数据集
    :param min_support: 最小支持度
    :return:
    """
    freq_items = {}  # 频繁项集
    for trans in data_set:  # 第一次遍历数据集
        for item in trans:
            freq_items[item] = freq_items.get(item, 0) + data_set[trans]

    header_table = {k: v for (k, v) in freq_items.items() if v >= min_support}  # 创建头指针表
    # for key in header_table:
    #     print key, header_table[key]

    # 无频繁项集
    if len(header_table) == 0:
        return None, None
    for k in header_table:
        header_table[k] = [header_table[k], None]  # 添加头指针表指向树中的数据
    # 创建树过程
    ret_tree = treeNode('Null Set', 1, None)  # 根节点

    # 第二次遍历数据集
    for trans, count in data_set.items():
        local_data = {}
        for item in trans:
            if header_table.get(item, 0):
                local_data[item] = header_table[item][0]
        if len(local_data) > 0:
            ##############################################################################################
            # 这里修改机器学习实战中的排序代码：
            ordered_items = [v[0] for v in sorted(local_data.items(), key=lambda kv: (-kv[1], kv[0]))]
            ##############################################################################################
            update_tree(ordered_items, ret_tree, header_table, count)  # populate tree with ordered freq itemset
    return ret_tree, header_table


def update_tree(items, in_tree, header_table, count):
    '''
    :param items: 元素项
    :param in_tree: 检查当前节点
    :param header_table:
    :param count:
    :return:
    '''
    if items[0] in in_tree.children:  # check if ordered_items[0] in ret_tree.children
        in_tree.children[items[0]].increase(count)  # incrament count
    else:  # add items[0] to in_tree.children
        in_tree.children[items[0]] = treeNode(items[0], count, in_tree)
        if header_table[items[0]][1] is None:  # update header table
            header_table[items[0]][1] = in_tree.children[items[0]]
        else:
            update_header(header_table[items[0]][1], in_tree.children[items[0]])
    if len(items) > 1:  # call update_tree() with remaining ordered items
        update_tree(items[1::], in_tree.children[items[0]], header_table, count)


def update_header(node_test, target_node):
    '''
    :param node_test:
    :param target_node:
    :return:
    '''
    while node_test.node_link is not None:  # Do not use recursion to traverse a linked list!
        node_test = node_test.node_link
    node_test.node_link = target_node


def ascend_tree(leaf_node, pre_fix_path):
    '''
    遍历父节点，找到路径
    :param leaf_node:
    :param pre_fix_path:
    :return:
    '''
    if leaf_node.parent is not None:
        pre_fix_path.append(leaf_node.name)
        ascend_tree(leaf_node.parent, pre_fix_path)


def find_pre_fix_path(base_pat, tree_node):
    '''
    创建前缀路径
    :param base_pat: 频繁项
    :param treeNode: FP树中对应的第一个节点
    :return:
    '''
    # 条件模式基
    cond_pats = {}
    while tree_node is not None:
        pre_fix_path = []
        ascend_tree(tree_node, pre_fix_path)
        if len(pre_fix_path) > 1:
            cond_pats[frozenset(pre_fix_path[1:])] = tree_node.count
        tree_node = tree_node.node_link
    return cond_pats


def mine_tree(in_tree, header_table, min_support, pre_fix, freq_items):
    '''
    挖掘频繁项集
    :param in_tree:
    :param header_table:
    :param min_support:
    :param pre_fix:
    :param freq_items:
    :return:
    '''
    # 从小到大排列table中的元素，为遍历寻找频繁集合使用
    bigL = [v[0] for v in sorted(header_table.items(), key=lambda p: p[0])]  # (sort header table)
    for base_pat in bigL:  # start from bottom of header table
        new_freq_set = pre_fix.copy()
        new_freq_set.add(base_pat)
        # print 'finalFrequent Item: ',new_freq_set    #append to set
        if len(new_freq_set) > 0:
            freq_items[frozenset(new_freq_set)] = header_table[base_pat][0]
        cond_patt_bases = find_pre_fix_path(base_pat, header_table[base_pat][1])
        my_cond_tree, my_head = create_tree(cond_patt_bases, min_support)
        # print 'head from conditional tree: ', my_head
        if my_head is not None:  # 3. mine cond. FP-tree
            # print 'conditional tree for: ',new_freq_set
            # my_cond_tree.disp(1)
            mine_tree(my_cond_tree, my_head, min_support, new_freq_set, freq_items)


def fp_growth(data_set, min_support=1):
    my_fp_tree, my_header_tab = create_tree(data_set, min_support)
    # my_fp_tree.disp()
    freq_items = {}
    mine_tree(my_fp_tree, my_header_tab, min_support, set([]), freq_items)
    return freq_items