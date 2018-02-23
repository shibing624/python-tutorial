# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: ltp demo
import os

ltp_data_dir = '/Users/xuming06/Codes/ltp_data_v3.4.0'

# segment
cws_model_path = os.path.join(ltp_data_dir, 'cws.model')

from pyltp import Segmentor

segmentor = Segmentor()
segmentor.load(cws_model_path)
text = '我是中国人，我在爱斯基摩打雪仗。欧几里得是西元前三世纪的希腊数学家'
words = segmentor.segment(text)
print(" ".join(words))  # 我 是 中国 人 ， 我 在 爱斯基摩 打雪仗 。 欧几 里 得 是 西元前 三 世纪 的 希腊 数学家

# segment with lexicon
segmentor = Segmentor()
# load self dictionary
segmentor.load_with_lexicon(cws_model_path, './self_dict.txt')
words = segmentor.segment(text)
print(" ".join(words))  # 我 是 中国 人 ， 我 在 爱斯基摩 打雪仗 。 欧几里得 是 西元前 三 世纪 的 希腊 数学家

# pos
pos_model_path = os.path.join(ltp_data_dir, 'pos.model')

from pyltp import Postagger

postagger = Postagger()
postagger.load(pos_model_path)
words = segmentor.segment(text)
print(' '.join(words))
postags = postagger.postag(words)
print(' '.join(postags))
zipped = zip(words, postags)
word_pos = list(i[0] + '/' + i[1] for i in zipped)
print(' '.join(word_pos))


# ner
ner_model_path = os.path.join(ltp_data_dir, 'ner.model')
from pyltp import NamedEntityRecognizer

recognizer = NamedEntityRecognizer()
recognizer.load(ner_model_path)
words = segmentor.segment(text)
postags = postagger.postag(words)
nertags = recognizer.recognize(words, postags)

print(' '.join(nertags))


# parser
par_model_path = os.path.join(ltp_data_dir, 'parser.model')
from pyltp import Parser

parser = Parser()
parser.load(par_model_path)

words = segmentor.segment(text)
postags = postagger.postag(words)
arcs = parser.parse(words, postags)

rely_id = [arc.head for arc in arcs]
relation = [arc.relation for arc in arcs]
heads = ['root' if id == 0 else words[id - 1] for id in rely_id]
for i in range(len(words)):
    print(relation[i] + '(' + words[i] + ', ' + heads[i] + ')')

# SRL
srl_model_path = os.path.join(ltp_data_dir,'pisrl.model')
from pyltp import SementicRoleLabeller
srl = SementicRoleLabeller()
srl.load(srl_model_path)
roles = srl.label(words,postags,arcs)
for role in roles:
    print(role.index, "".join(
        ["%s:(%d,%d)" % (arg.name, arg.range.start,arg.range.end) for arg in role.arguments]))

srl.release()

# jieba
import jieba
print(' '.join(jieba.cut(text)))

import jieba.posseg
words = jieba.posseg.cut(text)
word_pos = list(i + '/' + j for i,j in words)
print(' '.join(word_pos))