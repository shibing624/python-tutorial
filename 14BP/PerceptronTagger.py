# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 17/8/10
# Brief: 平均感知机：词性标注测试

import os
import random
from collections import defaultdict
import pickle
import logging

from AveragePerceptron import AveragePerceptron

PICKLE = "../data/bp/trontagger-0.1.pkg"
TRAIN_FILE_PATH = "../data/bp/train.txt"
TEST_FILE_PATH = "../data/bp/test.txt"


class PerceptronTagger():
    START = ['-START-', '-START2-']
    END = ['-END-', '-END2-']
    AP_MODEL_LOC = os.path.join(os.path.dirname(__file__), PICKLE)

    def __init__(self, load=True):
        self.model = AveragePerceptron()
        self.tagdict = {}
        self.classes = set()
        if load:
            self.load(self.AP_MODEL_LOC)

    def tag(self, corpus):
        s_split = lambda t: t.split('\n')
        w_split = lambda s: s.split()

        def split_sents(corpus):
            for s in s_split(corpus):
                yield w_split(s)

        prev, prev2 = self.START
        tokens = []
        for words in split_sents(corpus):
            context = self.START + [self._normalize(w) for w in words] + self.END
            for i, word in enumerate(words):
                tag = self.tagdict.get(word)
                if not tag:
                    features = self._get_features(i, word, context, prev, prev2)
                    tag = self.model.predict(features)
                tokens.append((word, tag))
                prev2 = prev
                prev = tag
        return tokens

    def load(self, loc):
        try:
            w_td_c = pickle.load(open(loc, 'rb'))
        except IOError:
            raise IOError("Missing trontagger.pkg file.")
        self.model.weights, self.tagdict, self.classes = w_td_c
        self.model.classes = self.classes
        return None

    def _normalize(self, word):
        if '-' in word and word[0] != '-':
            return '!HYPHEN'
        elif word.isdigit() and len(word) == 4:
            return '!YEAR'
        elif word[0].isdigit():
            return '!DIGITS'
        else:
            return word.lower()

    def _get_features(self, i, word, context, prev, prev2):
        i += len(self.START)
        features = defaultdict(int)

        def add(name, *args):
            features[' '.join((name,) + tuple(args))] += 1

        # constant feature
        add('bias')
        add('i suffix', word[-3:])
        add('i pref1', word[0])
        add('i-1 tag', prev)
        add('i-2 tag', prev2)
        add('i tag+i-2 tag', prev, prev2)
        add('i word', context[i])
        add('i-1 tag+i word', prev, context[i])
        add('i-1 word', context[i - 1])
        add('i-1 suffix', context[i - 1][-3:])
        add('i-2 word', context[i - 2])
        add('i+1 word', context[i + 1])
        add('i+1 suffix', context[i + 1][-3:])
        add('i+2 word', context[i + 2])
        return features

    def _make_tagdict(self, sentences):
        counts = defaultdict(lambda: defaultdict(int))
        for words, tags in sentences:
            for word, tag in zip(words, tags):
                counts[word][tag] += 1
                self.classes.add(tag)
        freq_thresh = 20
        ambiguity_thresh = 0.97
        for word, tag_freqs in counts.items():
            tag, mode = max(tag_freqs.items(), key=lambda item: item[1])
            n = sum(tag_freqs.values())
            if n >= freq_thresh and (float(mode) / n) >= ambiguity_thresh:
                self.tagdict[word] = tag

    def _pc(self, n, d):
        return (float(n) / d) * 100

    def train(self, sentences, save_loc=None, nr_iter=5):
        self._make_tagdict(sentences)
        self.model.classes = self.classes
        for iter_ in range(nr_iter):
            c = 0
            n = 0
            for words, tags in sentences:
                prev, prev2 = self.START
                context = self.START + [self._normalize(w) for w in words] + self.END
                for i, word in enumerate(words):
                    guess = self.tagdict.get(word)
                    if not guess:
                        feats = self._get_features(i, word, context, prev, prev2)
                        guess = self.model.predict(feats)
                        self.model.update(tags[i], guess, feats)
                    prev2 = prev
                    prev = guess
                    c += guess == tags[i]
                    n += 1
            random.shuffle(sentences)
            logging.info("Iter {0}: {1}/{2}={3}".format(iter_, c, n, self._pc(c, n)))
        self.model.average_weights()
        if save_loc is not None:
            pickle.dump((self.model.weights, self.tagdict, self.classes),
                        open(save_loc, 'wb'), -1)
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tagger = PerceptronTagger(False)
    try:
        tagger.load(PICKLE)
        print(tagger.tag("how are you ?"))
        logging.info("Start testing...")
        right = 0.0
        total = 0.0
        sentence = ([], [])
        for line in open(TEST_FILE_PATH):
            params = line.split()
            if len(params) != 2: continue
            sentence[0].append(params[0])
            sentence[1].append(params[1])
            if params[0] == ".":
                text = ""
                words = sentence[0]
                tags = sentence[1]
                for i, word in enumerate(words):
                    text += word
                    if i < len(words):
                        text += " "
                outputs = tagger.tag(text)
                assert len(tags) == len(outputs)
                total += len(tags)
                for o, t in zip(outputs, tags):
                    if o[1].strip() == t:
                        right += 1
                sentence = ([], [])
        logging.info("Precision : %f", right / total)
    except IOError:
        logging.info("Reading corpus...")
        training_data = []
        sentence = ([], [])
        for line in open(TRAIN_FILE_PATH):
            params = line.split('\t')
            sentence[0].append(params[0])
            sentence[1].append(params[1])
            if params[0] == ".":
                training_data.append(sentence)
                sentence = ([], [])
        logging.info("training corpus size: %d", len(training_data))
        logging.info("Start training...")
        tagger.train(training_data, save_loc=PICKLE)
        logging.info("training end.")
