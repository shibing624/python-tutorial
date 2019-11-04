# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""


def load_word_set(save_path):
    words = set()
    with open(save_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith("#"):
                continue
            words.add(line.strip())
    return words


def load_query_set(save_path):
    words = set()
    with open(save_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith("#"):
                continue
            words.add(line.strip())
    return words


class Risk(object):
    def __init__(self, risk_words_path=""):
        self.risk_words = load_word_set(risk_words_path)
        print("risk words size: %d" % len(self.risk_words))

    def check(self, query):
        # query is empty , return None
        if not query.strip():
            return
        for w in self.risk_words:
            if w in query:
                return w
        return None


if __name__ == "__main__":
    risk = Risk(risk_words_path="medical_tech.txt")
    input_queries = load_query_set("medical_tech_remark.txt")
    stat = []
    num = 0
    for i in input_queries:
        ret = risk.check(i)
        if ret:
            num += 1
            stat.append([ret, i[:10], num])
    print(stat)
