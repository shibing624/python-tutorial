# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

from chatterbot import ChatBot
from chatterbot.conversation import Statement
from chatterbot.trainers import ListTrainer
import logging
logging.basicConfig(level=logging.INFO)
"""
This example shows how to create a chat bot that
will learn responses based on an additional feedback
element from the user.
"""

# Uncomment the following line to enable verbose logging
# import logging
# logging.basicConfig(level=logging.INFO)

# Create a new instance of a ChatBot
bot = ChatBot(
    'Feedback Learning Bot',
    logic_adapters=[
        {
            'import_path': 'chatterbot.logic.BestMatch',
            'default_response': 'I am sorry, but I do not understand.',
            'maximum_similarity_threshold': 0.90
        }
    ],
    storage_adapter='chatterbot.storage.SQLStorageAdapter'
)

trainer = ListTrainer(bot)

trainer.train([
    "Hi, can I help you?",
    "Sure, I'd like to book a flight to Iceland.",
    "Your flight has been booked.",
    'hi',
    'hello',
    'what is your sex?',
    'female',
    'bye',
    'byebye'
])


def get_feedback():
    text = input()

    if 'yes' in text.lower() or 'y' in text.lower():
        return True
    elif 'no' in text.lower() or 'n' in text.lower():
        return False
    else:
        print('Please type either "yes" or "no"')
        return get_feedback()


print('Type something to begin...')

# The following loop will execute each time the user enters input
while True:
    try:
        input_statement = Statement(text=input())
        response = bot.get_response(input_statement)
        print(response)
        print('\n Is "{}" a right response to "{}"? \n'.format(
            response.text,
            input_statement.text
        ))
        if not get_feedback():
            print('please input the correct one')
            correct_response = Statement(text=input())
            bot.learn_response(correct_response, input_statement)
            print('Responses added to bot!')

    # Press ctrl-c or ctrl-d on the keyboard to exit
    except (KeyboardInterrupt, EOFError, SystemExit):
        break
