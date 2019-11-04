# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
import argparse
import logging
parser = argparse.ArgumentParser(description="starts the ask-question-robot")
parser.add_argument("task",
                    type=str,
                    choices=["train_nlu", "train_dialogue", "run", "run_online"],
                    help="what the bot should do - e.g. run or train?")
parser.add_argument("risk",
                    type=str,
                    choices=["gamble", "lottery", "mahjong"],
                    default="gamble",
                    help="what risk domain")
args = parser.parse_args()
print(args)