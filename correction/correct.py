# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 17/12/29
# Brief: correct main

import os
import kenlm
import jieba
import pickle
import math
import wubi
import numpy as np
from pypinyin import  pinyin
from collections import Counter
import config

print('Loading models...')
jieba.initialize()

bimodel_path = ""