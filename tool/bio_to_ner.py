# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

bio_index_dict = {
    'O': 0,
    'B-R': 1,
    'I-R': 2,
    'B-M': 3,
    'I-M': 4,
    'B-S': 5,
    'I-S': 6,
}
index_bio_dict = {str(v): k for k, v in bio_index_dict.items()}


def bio_to_ner(query_split, tag_split):
    res = []
    for id, tag in enumerate(tag_split):
        if 'R' in tag:
            res.append(query_split[id] + '_R')
        elif 'M' in tag:
            res.append(query_split[id] + '_M')
        elif 'S' in tag:
            res.append(query_split[id] + '_S')
    return res


def index_to_bio(tag_line):
    tag_line = tag_line.replace('[', '').replace(']', '')
    tags = tag_line.split(', ')
    tags = [index_bio_dict[i] for i in tags]
    return tags


if __name__ == '__main__':
    is_ori_index = True
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    with open(input_file, 'r', encoding='utf-8') as fr, \
        open(output_file, 'w', encoding='utf-8') as fw:
        for line in fr:
            line = line.strip()
            terms = line.split('\t')
            query_split = terms[0].split(' ')
            tag_line = terms[1]
            query = ''.join(query_split)
            if is_ori_index:
                tag_split = index_to_bio(tag_line)
            else:
                tag_split = terms[1].split(' ')
            if len(query_split) != len(tag_split):
                print('error size not match', len(query_split), len(tag_split))
                fw.write(query + '\t' + '\n')
                continue
            ners = bio_to_ner(query_split, tag_split)
            ners = ners if ners else []
            fw.write(query + '\t' + ' '.join(ners) + '\n')
