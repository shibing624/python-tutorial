# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 18/1/25
# Brief: 
import unittest
from util import load_load
from feature import Feature

train_file = "./data/train.data.sample"
test_file = "../data/train.data.sample"


class ClassificationTest(unittest.TestCase):
    """Test Case for classification
    """

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def test_init(self):
        print("test_init")
        """测试初始化函数，捕捉异常"""
        data_x, data_y = load_load(train_file)
        self.assertEqual(len(data_x) > 0, True)


if __name__ == '__main__':
    unittest.main()
