# -*- coding: utf-8 -*-
"""
@description: 情感分析
@author:XuMing
"""
from __future__ import print_function  # 兼容python3的print写法
from __future__ import unicode_literals  # 兼容python3的编码处理

import re, math, string
from itertools import product
from inspect import getsourcefile
from os.path import abspath, join, dirname

B_INCR = 0.293
B_DECR = -0.293

C_INCR = 0.733
N_SCALAR = -0.74

REGEX_REMOVE_PUNCTUATION = re.compile('[{0}]'.format(re.escape(string.punctuation)))
PUNC_LIST = [".", "!", "?", ",", ";", ":", "-", "'", "\"", "!!",
             "!!!", "??", "???", "?!?", "!?!", "?!?!", "!?!?"]
NEGATE = ['aint', 'arent', 'cannot', 'cant', 'couldnt', 'darent', 'didnt',
          "doesnt", "ain't", "aren't", "couldn't", "daren't", "didn't", "doesn't",
          "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't",
          "neednt", "needn't", "never", "none", "nope", "nor", "not", "nothing", "nowhere",
          "oughtnt", "shant", "shouldnt", "uhuh", "wasnt", "werent",
          "oughtn't", "shan't", "shouldn't", "uh-uh", "wasn't", "weren't",
          "without", "wont", "wouldnt", "won't", "wouldn't", "rarely", "seldom", "despite"]

BOOSTER_DICT = \
    {"absolutely": B_INCR, "amazingly": B_INCR, "awfully": B_INCR, "completely": B_INCR, "considerably": B_INCR,
     "decidedly": B_INCR, "deeply": B_INCR, "effing": B_INCR, "enormously": B_INCR,
     "entirely": B_INCR, "especially": B_INCR, "exceptionally": B_INCR, "extremely": B_INCR,
     "fabulously": B_INCR, "flipping": B_INCR, "flippin": B_INCR,
     "fricking": B_INCR, "frickin": B_INCR, "frigging": B_INCR, "friggin": B_INCR, "fully": B_INCR, "fucking": B_INCR,
     "greatly": B_INCR, "hella": B_INCR, "highly": B_INCR, "hugely": B_INCR, "incredibly": B_INCR,
     "intensely": B_INCR, "majorly": B_INCR, "more": B_INCR, "most": B_INCR, "particularly": B_INCR,
     "purely": B_INCR, "quite": B_INCR, "really": B_INCR, "remarkably": B_INCR,
     "so": B_INCR, "substantially": B_INCR,
     "thoroughly": B_INCR, "totally": B_INCR, "tremendously": B_INCR,
     "uber": B_INCR, "unbelievably": B_INCR, "unusually": B_INCR, "utterly": B_INCR,
     "very": B_INCR,
     "almost": B_DECR, "barely": B_DECR, "hardly": B_DECR, "just enough": B_DECR,
     "kind of": B_DECR, "kinda": B_DECR, "kindof": B_DECR, "kind-of": B_DECR,
     "less": B_DECR, "little": B_DECR, "marginally": B_DECR, "occasionally": B_DECR, "partly": B_DECR,
     "scarcely": B_DECR, "slightly": B_DECR, "somewhat": B_DECR,
     "sort of": B_DECR, "sorta": B_DECR, "sortof": B_DECR, "sort-of": B_DECR}

# check for special case idioms using a sentiment-laden keyword known to VADER
SPECIAL_CASE_IDIOMS = {"the shit": 3, "the bomb": 3, "bad ass": 1.5, "yeah right": -2,
                       "cut the mustard": 2, "kiss of death": -1.5, "hand to mouth": -2}


def negated(input_words, include_nt=True):
    """
    Determine if input contains negation words
    :param input_words: 
    :param include_nt: 
    :return: 
    """
    neg_words = []
    neg_words.extend(NEGATE)
    for word in neg_words:
        if word in input_words:
            return True
    if include_nt:
        for word in input_words:
            if "n't" in word:
                return True
    if "least" in input_words:
        i = input_words.index("least")
        if i > 0 and input_words[i - 1] != "at":
            return True
    return False


def normalize(score, alpha=15):
    """
    Normalize the score to be between -1 and 1 using an alpha
    :param score: 
    :param alpha: 
    :return: 
    """
    norm_score = score / math.sqrt((score * score) + alpha)
    if norm_score < -1.0:
        return -1.0
    elif norm_score > 1.0:
        return 1.0
    else:
        return norm_score


def allcap_differential(words):
    """
    Check the input are ALL CAP
    :param words: list words
    :return: "TURE" if some but not all words all ALL CAPS
    """
    is_different = False
    allcap_words = 0
    for word in words:
        if word.isupper():
            allcap_words += 1
    cap_differential = len(words) - allcap_words
    if 0 < cap_differential < len(words):
        is_different = True
    return is_different


def scalar_inc_dec(word, valence, is_cap_diff):
    """
    Check if the preceding words increase,decrease, or negate/nullify the valence
    :param word: 
    :param valence: 
    :param is_cap_diff: 
    :return: 
    """
    scalar = 0.0
    word_lower = word.lower()
    if word_lower in BOOSTER_DICT:
        scalar = BOOSTER_DICT[word_lower]
        if valence < 0:
            scalar *= -1
        # check if word in ALL CAPS
        if word.isupper() and is_cap_diff:
            if valence > 0:
                scalar += C_INCR
            else:
                scalar -= C_INCR
    return scalar


class SentiText(object):
    """
    Identify sentiment-relevant string-level properties of input text.
    """

    def __init__(self, text):
        if not isinstance(text, str):
            text = str(text.encode('utf-8'))
        self.text = text
        self.words_and_emoticons = self._words_and_emoticons()
        self.is_cap_diff = allcap_differential(self.words_and_emoticons)

    def _words_plus_punc(self):
        """
        get word and punctuation
        :return: mapping of form:
        {
        'cat,':'cat',
        ',cat':'cat',
        }
        """
        no_punc_text = REGEX_REMOVE_PUNCTUATION.sub('', self.text)
        # removes punctuation
        words_only = no_punc_text.split()
        # remove singletons
        words_only = set(w for w in words_only if len(w) > 1)
        # the product gives ('cat',',')
        punc_before = {''.join(p): p[1] for p in product(PUNC_LIST, words_only)}
        punc_after = {''.join(p): p[0] for p in product(words_only, PUNC_LIST)}
        words_punc_dict = punc_before
        words_punc_dict.update(punc_after)
        return words_punc_dict

    def _words_and_emoticons(self):
        """
        Remove leading and trailing punctuation
        Leave emoticons
        :return: 
        """
        wes = self.text.split()
        words_punc_dict = self._words_plus_punc()
        wes = [we for we in wes if len(we) > 1]
        for i, we in enumerate(wes):
            if we in words_punc_dict:
                wes[i] = words_punc_dict[we]
        return wes


class SentimentIntensityAnalyzer(object):
    """
    Give a sentiment intensity score to sentences.
    """

    def __init__(self, lexicon_file="lexicon.txt"):
        _this_module_file_path_ = abspath(getsourcefile(lambda: 0))
        lexicon_full_file_path = join(dirname(_this_module_file_path_), lexicon_file)
        with open(lexicon_full_file_path) as f:
            self.lexicon_full_file_path = f.read()
        self.lexicon = self.make_lex_dict()

    def make_lex_dict(self):
        """
        Convert lexicon file to a dictionary
        :return: 
        """
        lex_dict = {}
        for line in self.lexicon_full_file_path.split('\n'):
            (word, measure) = line.strip().split('\t')[0:2]
            lex_dict[word] = float(measure)
        return lex_dict

    def polarity_scores(self, text):
        """
        Return a float for sentiment strength based on the input text.
        :param text: 
        :return: positive values ,negative value 
        """
        sentitext = SentiText(text)

        sentiments = []
        words_and_emoticons = sentitext.words_and_emoticons
        for item in words_and_emoticons:
            valence = 0
            i = words_and_emoticons.index(item)
            if (i < len(words_and_emoticons) - 1 and item.lower() == "kind" and \
                        words_and_emoticons[i + 1].lower() == "of") or \
                    item.lower() in BOOSTER_DICT:
                sentiments.append(valence)
                continue

            sentiments = self.sentiment_valence(valence, sentitext, item, i, sentiments)
        sentiments = self._but_check(words_and_emoticons, sentiments)
        valence_dict = self.score_valence(sentiments, text)
        return valence_dict

    def sentiment_valence(self, valence, sentitext, item, i, sentiments):
        is_cap_diff = sentitext.is_cap_diff
        words_and_emoticons = sentitext.words_and_emoticons
        item_lowercase = item.lower()
        if item_lowercase in self.lexicon:
            valence = self.lexicon[item_lowercase]
            if item.isupper() and is_cap_diff:
                if valence > 0:
                    valence += C_INCR
                else:
                    valence -= C_INCR
            for start_i in range(0, 3):
                if i > start_i and words_and_emoticons[i - (start_i + 1)].lower() \
                    not in self.lexicon:
                    s = scalar_inc_dec(words_and_emoticons[i - (start_i + 1)], \
                                       valence, is_cap_diff)
                    if start_i == 1 and s != 0:
                        s = s * 0.95
                    if start_i == 2 and s != 0:
                        s = s * 0.9
                    valence = valence + s
                    valence = self._never_check(valence, words_and_emoticons, start_i, i)
                    if start_i == 2:
                        valence = self._idioms_check(valence, words_and_emoticons, i)

            valence = self._least_check(valence, words_and_emoticons, i)
        sentiments.append(valence)
        return sentiments

    def _least_check(self, valence, words_and_emoticons, i):
        """
        Check for negation case
        :param valence: 
        :param words_and_emoticons: 
        :param i: 
        :return: 
        """
        if i > 1 and words_and_emoticons[i - 1].lower() not in self.lexicon \
            and words_and_emoticons[i - 1].lower() == "least":
            if words_and_emoticons[i - 2].lower() != "at" and \
                    words_and_emoticons[i - 2].lower() != "very":
                valence = valence * N_SCALAR
        elif i > 0 and words_and_emoticons[i - 1].lower() not in self.lexicon and \
                words_and_emoticons[i - 1].lower() == "least":
            valence = valence * N_SCALAR
        return valence

    def _but_check(self, words_and_emoticons, sentiments):
        if 'but' in words_and_emoticons or 'BUT' in words_and_emoticons:
            try:
                bi = words_and_emoticons.index('but')
            except ValueError:
                bi = words_and_emoticons.index('BUT')
            for sentiment in sentiments:
                si = sentiments.index(sentiment)
                if si < bi:
                    sentiments.pop(si)
                    sentiments.insert(si, sentiment * 0.5)
                elif si > bi:
                    sentiments.pop(si)
                    sentiments.insert(si, sentiment * 1.5)
        return sentiments

    def _idioms_check(self, valence, words_and_emoticons, i):
        onezero = "{0} {1}".format(words_and_emoticons[i - 1], words_and_emoticons[i])
        twoonezero = "{0} {1} {2}".format(words_and_emoticons[i - 2], words_and_emoticons[i - 1],
                                          words_and_emoticons[i])
        twoone = "{0} {1}".format(words_and_emoticons[i - 2], words_and_emoticons[i - 1])
        threetwoone = "{0} {1} {2}".format(words_and_emoticons[i - 3], words_and_emoticons[i - 2],
                                           words_and_emoticons[i - 1])
        threetwo = "{0} {1}".format(words_and_emoticons[i - 3], words_and_emoticons[i - 2])
        sequences = [onezero, twoonezero, twoone, threetwoone, threetwo]
        for seq in sequences:
            if seq in SPECIAL_CASE_IDIOMS:
                valence = SPECIAL_CASE_IDIOMS[seq]
                break
        if len(words_and_emoticons) - 1 > i:
            zeroone = "{0} {1}".format(words_and_emoticons[i], words_and_emoticons[i + 1])
            if zeroone in SPECIAL_CASE_IDIOMS:
                valence = SPECIAL_CASE_IDIOMS[zeroone]
        if len(words_and_emoticons) - 1 > i + 1:
            zeroonetwo = "{0} {1} {2}".format(words_and_emoticons[i], words_and_emoticons[i + 1],
                                              words_and_emoticons[i + 2])
            if zeroonetwo in SPECIAL_CASE_IDIOMS:
                valence = SPECIAL_CASE_IDIOMS[zeroonetwo]
        # Check for booster/dampener bi-grams such as 'sort of' or 'kind of'
        if threetwo in BOOSTER_DICT or twoone in BOOSTER_DICT:
            valence = valence + B_DECR
        return valence

    def _never_check(self, valence, words_and_emoticons, start_i, i):
        if start_i == 0:
            if negated([words_and_emoticons[i - 1]]):
                valence = valence * N_SCALAR
        if start_i == 1:
            if words_and_emoticons[i - 2] == "never" and \
                (words_and_emoticons[i - 1] == "so" or words_and_emoticons[i - 1] == "this"):
                valence = valence * 1.5
            elif negated([words_and_emoticons[i - (start_i + 1)]]):
                valence = valence * N_SCALAR
        if start_i == 2:
            if words_and_emoticons[i - 3] == "never" and \
                (words_and_emoticons[i - 2] == "so" or words_and_emoticons[i - 2] == "this") or \
                (words_and_emoticons[i - 1] == "so" or words_and_emoticons[i - 1] == "this"):
                valence = valence * 1.25
            elif negated([words_and_emoticons[i - (start_i + 1)]]):
                valence = valence * N_SCALAR
        return valence

    def _punctuation_emphasis(self, text):
        # add emphasis
        ep_amplifier = self._amplify_ep(text)
        qm_amplifier = self._amplify_qm(text)
        punc_emph_amplifier = ep_amplifier + qm_amplifier
        return punc_emph_amplifier

    def _amplify_qm(self, text):
        qm_count = text.count("?")
        qm_amplifier = 0;
        if qm_count > 1:
            if qm_count <= 3:
                qm_amplifier = qm_count * 0.18
            else:
                qm_amplifier = 0.96
        return qm_amplifier

    def _amplify_ep(self, text):
        # check for added emphasis
        ep_count = text.count("!")
        if ep_count > 4:
            ep_count = 4
        ep_amplifier = ep_count * 0.292
        return ep_amplifier

    def _sift_sentiment_scores(self, sentiments):
        # separate positive versus negative sentiment scores
        pos_sum = 0.0
        neg_sum = 0.0
        neu_count = 0
        for sentiment_score in sentiments:
            if sentiment_score > 0:
                pos_sum += (float(sentiment_score) + 1)
            if sentiment_score < 0:
                neg_sum += (float(sentiment_score) - 1)
            if sentiment_score == 0:
                neu_count += 1
        return pos_sum, neg_sum, neu_count

    def score_valence(self, sentiments, text):
        if sentiments:
            sum_s = float(sum(sentiments))
            punct_emph_amplifier = self._punctuation_emphasis( text)
            if sum_s > 0:
                sum_s += punct_emph_amplifier
            elif sum_s < 0:
                sum_s -= punct_emph_amplifier

            compound = normalize(sum_s)
            # discriminate between positive, negative and neutral sentiment scores
            pos_sum, neg_sum, neu_count = self._sift_sentiment_scores(sentiments)
            if pos_sum > math.fabs(neg_sum):
                pos_sum += (punct_emph_amplifier)
            elif pos_sum < math.fabs(neg_sum):
                neg_sum -= (punct_emph_amplifier)

            total = pos_sum + math.fabs(neg_sum) + neu_count
            pos = math.fabs(pos_sum / total)
            neg = math.fabs(neg_sum / total)
            neu = math.fabs(neu_count / total)

        else:
            compound = 0.0
            pos = 0.0
            neg = 0.0
            neu = 0.0
        sentiment_dict = {"neg": round(neg, 3),
                          "neu": round(neu, 3),
                          "pos": round(pos, 3),
                          "compound": round(compound, 4)}
        return sentiment_dict
