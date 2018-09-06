# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 17/9/7
# Brief: 
# !/usr/bin/env python
# -*- coding: gb18030 -*-
########################################################################
#
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
#
########################################################################

import sys
import re
reload(sys)
sys.setdefaultencoding('gb18030')
import urllib

TIGHT_THRESHOLD = 0.7


def word_tight_init():
    """
    初始化word_tight
    """
    sofa.use('drpc.ver_1_0_0', 'S')
    sofa.use('nlpc.ver_1_0_0', 'wordtight')

    conf = sofa.Config()
    conf.load('./config/drpc_client.xml')

    return S.ClientAgent(conf['sofa.service.nlpc_wordtight_124'])


def word_tight_seg(wordtight_agent, source_str):
    """
    利用word_tight组合分词结果
    """
    seg_list = []

    if len(source_str) <= 0:
        return None

    source_str = source_str.strip()
    m_input = wordtight.wordtight_input()
    m_input.query = str(source_str)

    input_data = sofa.serialize(m_input)

    for i in range(5):
        try:
            ret, output_data = wordtight_agent.call_method(input_data)
            break
        except Exception as e:
            continue

    if len(output_data) == 0:
        print >> sys.stderr, "No result!"
        return None

    m_output = wordtight.wordtight_output()
    m_output = sofa.deserialize(output_data, type(m_output))
    m_output = m_output.result

    stack = []
    stack.append(m_output.chkcnt - 1)
    flag = [0 for i in range(m_output.chkcnt)]

    while len(stack) > 0:
        chk = stack.pop()

        if flag[chk] == 0:
            flag[chk] = 1

        # 加入叶子节点，有时word_tight叶子节点紧密度不为1
        if m_output.chunks[chk].sub_chunk_count == 0:
            str_buff = "".join(m_output.chunks[chk].terms)
            seg_list.append(str_buff)
            continue

        # 如果短语紧密度大于指定阈值直接输出，不再递归其子节点
        if m_output.chunks[chk].tight > TIGHT_THRESHOLD:
            str_buff = "".join(m_output.chunks[chk].terms)
            seg_list.append(str_buff)
            continue

        for i in range(m_output.chunks[chk].sub_chunk_count)[::-1]:
            stack.append(m_output.chunks[chk].sub_chunk_indice[i])

    return seg_list


def gen_ngram(source_list, N=3):
    ngram_list = source_list
    length = len(source_list)

    for i in range(2, N):
        begin = 0
        while begin + i < length:
            ngram_list.append("".join(source_list[begin:begin + i]))
            begin += i

    return ngram_list


def load_comb_rule_vocabs(comb_rule_path):
    # 加载 gb18030 编码词典
    word_set = set()
    with open(comb_rule_path) as f:
        for line in f:
            word_set.add(line.strip().lower().decode("gb18030", errors="ignore"))
    return word_set


def load_comb_rule_vocabs2(comb_rule_path):
    # 加载 gb18030 编码词典
    word_set = []
    with open(comb_rule_path) as f:
        for line in f:
            rules = line.strip().lower().decode('gb18030', errors='ignore') \
                .encode('gb18030', errors='ignore').split("\t")
            # print "rules is ",rules
            word_set.append(rules)
            # print "word_set is ",word_set
    return word_set


def load_comb_rule_vocabs1(comb_rule_path):
    # 加载 utf8 编码词典
    word_set = set()
    with open(comb_rule_path) as f:
        for line in f:
            word_set.add(line.strip().lower().decode('utf8', errors='ignore'))  # %规则词表utf8编码格式
    return word_set


def check_rule(rule_set, source_str):
    """
    直接整句匹配
    """
    source_str = source_str.strip()  # .decode("gb18030")
    for rule in rule_set:
        if "," in rule:
            rule_list = rule.split(',')
            rule_offset = [source_str.find(i) for i in rule_list]
            if -1 not in rule_offset:
                dist = max(rule_offset) - min(rule_offset)
                if dist < 7:
                    dist1 = (max(rule_offset) + min(rule_offset)) / 2
                    return True, rule, dist1
        else:
            rule_offset = source_str.find(rule)
            if rule_offset != -1:
                return True, rule, rule_offset
    return False, "None", -1


def check_rule_pro(rule, source_list):
    """
    将句子用word_tight分词后匹配
    """
    if source_list == None:
        return True
    if "," in rule:
        rule_list = rule.split(",")
        rule_ret = [True for i in rule_list if i in source_list]
        return len(rule_ret) == len(rule_list)
    else:
        return rule in source_list


def check_rate1(sent):
    # func:判断文本是否包含**.*%
    rate = re.findall('[1]?[09][0-9]\.?[0-9]*%', sent)
    sent_len = len(sent)
    if len(rate) > 0:
        return True
    else:
        return False


def check_RateRule(rule_set, source_str):
    source_str = source_str.strip()  # .decode("utf8", errors="ignore")
    contain_tag = check_rate1(source_str)
    if contain_tag:
        for rule in rule_set:
            if "," in rule:
                rule_list = rule.split(',')
                rule_offset = [source_str.find(i) for i in rule_list]
                if -1 not in rule_offset:
                    dist = max(rule_offset) - min(rule_offset)
                    if dist < 7:
                        return True, rule
            else:
                rule_offset = source_str.find(rule)
                if rule_offset != -1:
                    return True, rule
    return False, "None"


def main():
    # wordseg_agent = wordseg_init()
    rule_set1 = load_comb_rule_vocabs("./data/Inputdata/jjy.rule.txt")  # 规则库1，金融教育医疗行业，gb18030格式，转为unicode
    rule_set2 = load_comb_rule_vocabs("./data/Inputdata/quanhangye.rule.txt")  # 规则库3,其他行业，gb18030格式，转为unicode
    rule_set3 = load_comb_rule_vocabs("./data/Inputdata/out.policy.txt")  # 其他行业政策内标准，gb18030格式，转为unicode
    rule_yw_set = load_comb_rule_vocabs("./data/Inputdata/filter.yw")  # 疑问词表，，gb18030格式，转为unicode
    rate_rule_set_inpolicy = load_comb_rule_vocabs(
        "./data/Inputdata/rate_mis_rule_inpolicy.txt")  # 规则库2，utf8格式，转为unicode
    rate_rule_set_outpolicy = load_comb_rule_vocabs(
        "./data/Inputdata/rate_mis_rule_outpolicy.txt")  # 规则库2，utf8格式，转为unicode

    path_official_variant = "./data/illegal/official_variant.txt"
    path_repair_variant = "./data/illegal/repair_variant.txt"
    path_symbol = "./data/illegal/symbol.txt"
    path_official_url = "./data/illegal/official_url.txt"
    path_clock = "./data/illegal/clock.txt"
    path_repair = "./data/illegal/repair.txt"

    official_variant_list = []
    with open(path_official_variant, "r")as fd:
        for line in fd:
            line = line.decode('gb18030')
            parts = line.strip().split("\t")
            word = parts[0]
            bt = parts[1]
            official_variant_list.append([word, bt])

    repair_variant_list = []
    with open(path_repair_variant, "r")as fd:
        for line in fd:
            line = line.decode('gb18030')
            parts = line.strip().split("\t")
            word = parts[0]
            bt = parts[1]
            repair_variant_list.append([word, bt])

    symbol_set = set()
    with open(path_symbol, "r")as fd:
        for line in fd:
            symbol_set.add(line.strip("\n").decode('gb18030', 'ignore'))

    url_set = set()
    with open(path_official_url, "r")as fd:
        for line in fd:
            url_set.add(line.strip().decode("utf-8"))

    clock_set = set()
    with open(path_clock, "r") as f:
        for i in f:
            clock_set.add(i.strip().decode("gb18030", "ignore"))

    repair_set = set()
    with open(path_repair, "r") as f:
        for i in f:
            repair_set.add(i.strip().decode("gb18030", "ignore"))

    for line in sys.stdin:
        if not len(line):
            continue
        parts = line.strip().split('\t')
        # shw = parts[0]
        # clk = parts[1]
        # price = parts[2]
        userid = parts[3]
        title = parts[7]
        desc1 = parts[8]
        desc2 = parts[9]
        showurl = parts[10]
        target_url = parts[11]
        text = title + desc1 + desc2
        text_lower = urllib.unquote(text).strip().lower().decode('gb18030', 'ignore')
        sents = re.split(u'\;|\；|\:|\：|\，|\,|\。\.|\！|\!|\?|\？|\、', text_lower)
        output_tag = 0  # 是否有风险的最终标识
        outpolicy_tag = 0
        ratetag = 0
        quanhangye_tag = 0
        jjy_tag = 0
        # print "".join(sents)
        global rule
        for sent in sents:
            if len(sent) == 0:
                continue
            hard_ret_yw, rule_yw, offset_yw = check_rule(rule_yw_set, sent)
            if hard_ret_yw:  # 疑问句
                continue
            rate_tag_inpolicy, rule = check_RateRule(rate_rule_set_inpolicy, sent)
            if rate_tag_inpolicy:
                output_tag = 1
                ratetag_inpolicy = 1
                break
            else:
                hard_tag, rule, offset1 = check_rule(rule_set2, sent)  ##全行业匹配
                if hard_tag:
                    output_tag = 1
                    quanhangye_tag = 1
                    break
                else:
                    hard_tag, rule, offset2 = check_rule(rule_set1, sent)  # 教育金融医疗匹配规则库1
                    if hard_tag:
                        output_tag = 1
                        jjy_tag = 1
                        break
                    else:
                        hard_tag, rule, offset2 = check_rule(rule_set3, sent)  # 政策外匹配
                        rate_tag_outpolicy, rule = check_RateRule(rate_rule_set_outpolicy, sent)  # 政策外 % 匹配
                        if hard_tag or rate_tag_outpolicy:
                            output_tag = 1
                            outpolicy_tag = 1
                            break
        text = text.decode('gb18030', 'ignore')
        for token in symbol_set:
            text = text.replace(token, "")

        official_variant_flag = False
        for k in official_variant_list:
            if k in text:
                official_variant_flag = True
                break

        repair_flag = False
        for k in repair_set:
            if k in text:
                repair_flag = True
                break

        url_flag = True
        for url in url_set:
            if (url in showurl) or (url in target_url):
                url_flag = False
                break

        official_rule = u""
        repair_rule = u""
        if repair_flag and url_flag and official_variant_flag:
            print("\t".join([line.strip(), str(output_tag), str(outpolicy_tag), rule,
                             (u'official_rule:' + official_rule).encode('gb18030'),
                             (u'repair_rule:' + repair_rule).encode('gb18030')]))


if __name__ == '__main__':
    main()
