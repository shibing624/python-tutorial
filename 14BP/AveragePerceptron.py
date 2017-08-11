# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 17/8/10
# Brief: 平均感知机

from collections import defaultdict
import pickle
import random


class AveragePerceptron:
    def __init__(self):
        self.weights = {}
        self.classes = set()
        self._totals = defaultdict(int)
        self._tstamps = defaultdict(int)
        self.i = 0

    def predict(self, features):
        scores = defaultdict(float)
        for feat, value in features.items():
            if feat not in self.weights or value == 0:
                continue
            weights = self.weights[feat]
            for label, weight in weights.items():
                scores[label] += value * weight
        return max(self.classes, key=lambda label: (scores[label], label))

    def update(self, truth, guess, features):
        """Update the feature weights"""

        def update_feat(c, f, w, v):
            param = (f, c)
            self._totals[param] += (self.i - self._tstamps[param]) * w
            self._tstamps[param] = self.i
            self.weights[f][c] = w + v

        self.i += 1
        if truth == guess:
            return None
        for f in features:
            weights = self.weights.setdefault(f, {})
            update_feat(truth, f, weights.get(truth, 0.0), 1.0)
            update_feat(guess, f, weights.get(guess, 0.0), -1.0)
        return None

    def average_weights(self):
        """Average weights from all iterator"""
        for feat, weights in self.weights.items():
            new_feat_weights = {}
            for clas, weight in weights.items():
                new_feat_weights = {}
                for clas, weight in weights.items():
                    param = (feat, clas)
                    total = self._totals[param]
                    total += (self.i - self._tstamps[param]) * weight
                    averaged = round(total / float(self.i), 3)
                    if averaged:
                        new_feat_weights[clas] = averaged
                self.weights[feat] = new_feat_weights
        return None

    def save(self, path):
        return pickle.dump(dict(self.weights), open(path, 'w'))

    def load(self, path):
        self.weights = pickle.load(open(path))
        return None


def train(nr_iter, examples):
    model = AveragePerceptron()
    for i in range(nr_iter):
        random.shuffle(examples)
        for features, clazz in examples:
            scores = model.predict(features)
            guess, score = max(scores.items(), key=lambda i: i[1])
            if guess != clazz:
                model.update(clazz, guess, features)
    model.average_weights()
    return model
