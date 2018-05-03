# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 
import time
import datetime
import xlearn as xl
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load dataset
iris_data = load_iris()
X = iris_data['data']
y = (iris_data['target'] == 2)

X_train, \
X_val, \
y_train, \
y_val = train_test_split(X, y, test_size=0.3, random_state=0)

# xlearn
# param:
#  0. binary classification
#  1. model scale: 0.1
#  2. epoch number: 10 (auto early-stop)
#  3. learning rate: 0.1
#  4. regular lambda: 1.0
#  5. use sgd optimization method
linear_model = xl.LRModel(task='binary', init=0.1,
                          epoch=10, lr=0.1,
                          reg_lambda=1.0, opt='sgd')
print(datetime.datetime.now())
# Start to train
linear_model.fit(X_train, y_train,
                 eval_set=[X_val, y_val],
                 is_lock_free=False)

# print model weights
# print(linear_model.weights)

# Generate predictions
y_pred = linear_model.predict(X_val)
print(datetime.datetime.now())
# sklearn
lr = LogisticRegression()
lr.fit(X_train, y_train)
print('{0}, val mean acc:{1}'.format(lr.__str__(), lr.score(X_val, y_val)))
print(datetime.datetime.now())