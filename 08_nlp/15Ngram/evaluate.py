from config import test_result_path
from config import test_gold_path


class Evaluate():
    def __init__(self):
        pass

    def evaluate(self):
        test_result_file = open(test_result_path, encoding='utf-8')
        test_gold_file = open(test_gold_path, encoding='utf-8')

        result_cnt = 0.0
        gold_cnt = 0.0
        right_cnt = 0.0

        for line1, line2 in zip(test_result_file, test_gold_file):
            result_list = line1.strip().split(' ')
            gold_list = line2.strip().split(' ')
            for words in gold_list:
                if words != '':
                    gold_list.remove(words)
            for words in gold_list:
                if words != '':
                    result_list.remove(words)

            result_cnt += len(result_list)
            gold_cnt += len(gold_list)
            for words in result_list:
                if words in gold_list:
                    right_cnt += 1.0
                    gold_list.remove(words)

        p = right_cnt / result_cnt
        r = right_cnt / gold_cnt
        F = 2.0 * p * r / (p + r + 1)

        print('right_cnt: \t\t', right_cnt)
        print('result_cnt: \t', result_cnt)
        print('gold_cnt: \t\t', gold_cnt)

        print('P: \t\t', p)
        print('R: \t\t', r)
        print('F: \t\t', F)


if __name__ == '__main__':
    E = Evaluate()
    E.evaluate()
