# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 
import os
import tarfile

import keras

from babi.util import get_stories
from babi.util import vectorize_stories

BATCH_SIZE = 32
save_model_path = 'babi_rnn_model.h5'
challenge = 'tasks_1-20_v1-2/en/qa2_two-supporting-facts_{}.txt'
pwd_path = os.path.abspath(os.path.dirname(__file__))
print('pwd_path:', pwd_path)
path = os.path.join(pwd_path, '../../data/babi_tasks_1-20_v1-2.tar.gz')
print('path:', path)
with tarfile.open(path) as tar:
    train = get_stories(tar.extractfile(challenge.format('train')))
    test = get_stories(tar.extractfile(challenge.format('test')))

vocab = set()
for story, q, a in train + test:
    vocab |= set(story + q + [a])
vocab = sorted(vocab)

vocab_size = len(vocab) + 1
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
story_maxlen = max(map(len, (x for x, _, _ in train + test)))
query_maxlen = max(map(len, (x for _, x, _ in train + test)))

test_idx_story, test_idx_query, test_idx_answer = vectorize_stories(test, word_idx, story_maxlen, query_maxlen)
# load model by file
model = keras.models.load_model(save_model_path)
probs = model.predict([test_idx_story, test_idx_query], batch_size=BATCH_SIZE)
assert len(probs) == len(test_idx_answer)
for answer, prob in zip(test_idx_answer, probs):
    print('answer_test_index:%s\tprob_index:%s\tprob:%s' % (answer, prob.argmax(), prob.max()))
print(probs)
print(test_idx_answer)
