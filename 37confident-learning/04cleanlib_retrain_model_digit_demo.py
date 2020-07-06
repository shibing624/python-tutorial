# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

# # simplified Confident Learning Tutorial
# *Author: Curtis G. Northcutt, cgn@mit.edu*
#
# In this tutorial, we show how to implement confident learning without using cleanlab (for the most part).
# This tutorial is to confident learning what this tutorial https://pytorch.org/tutorials/beginner/examples_tensor/two_layer_net_numpy.html
# is to deep learning.
#
# The actual implementations in cleanlab are complex because they support parallel processing, numerous type and input checks, lots of hyper-parameter settings, lots of utilities to make things work smoothly for all types of inputs, and ancillary functions.
#
# I ignore all of that here and provide you a bare-bones implementation using mostly for-loops and some numpy.
# Here we'll do two simple things:
# 1. Compute the confident joint which fully characterizes all label noise.
# 2. Find the indices of all label errors, ordered by likelihood of being an error.
#
# ## INPUT (stuff we need beforehand):
# 1. s - These are the noisy labels. This is an np.array of noisy labels, shape (n,1)
# 2. psx - These are the out-of-sample holdout predicted probabilities for every example in your dataset. This is an np.array (2d) of probabilities, shape (n, m)
#
# ## OUTPUT (what this returns):
# 1. confident_joint - an (m, m) np.array matrix characterizing all the label error counts for every pair of labels.
# 2. label_errors_idx - a numpy array comprised of indices of every label error, ordered by likelihood of being a label error.
#
# In this tutorial we use the handwritten digits dataset as an example.

# In[1]:


from __future__ import print_function, absolute_import, division, with_statement

# To silence convergence warnings caused by using a weak
# logistic regression classifier on image data
import warnings

import cleanlab
import numpy as np
from cleanlab.classification import LearningWithNoisyLabels
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

warnings.simplefilter("ignore")
np.random.seed(477)

# In[2]:


# STEP 0 - Get some real digits data. Add a bunch of label errors. Get probs.

# Get handwritten digits data
X = load_digits()['data']
y = load_digits()['target']
print("X:", X[:10])
print("y:", y[:100])
print('Handwritten digits datasets number of classes:', len(np.unique(y)))
print('Handwritten digits datasets number of examples:', len(y))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
# Add lots of errors to labels
s = np.array(y_train)
for i in range(100):
    # Switch to some wrong label thats a different class
    s[i] = 0

# Confirm that we indeed added NUM_ERRORS label errors
actual_label_errors = np.arange(len(y_train))[s != y_train]
print('\nIndices of actual label errors:\n', actual_label_errors)
print('error with y, y[:20]:', s[:20])
print("len of errors:", len(actual_label_errors))
actual_num_errors = len(actual_label_errors)
# To keep the tutorial short, we use cleanlab to get the
# out-of-sample predicted probabilities using cross-validation
# with a very simple, non-optimized logistic regression classifier
clf = LogisticRegression()
psx = cleanlab.latent_estimation.estimate_cv_predicted_probabilities(
    X_train, s, clf=clf)

# Now we have our noisy labels s and predicted probabilities psx.
# That's all we need for confident learning.


# STEP 1 - Compute confident joint

# Verify inputs
s = np.asarray(s)
psx = np.asarray(psx)

# Find the number of unique classes if K is not given
K = len(np.unique(s))

from cleanlab.pruning import get_noise_indices

ordered_label_errors = get_noise_indices(
    s=s,
    psx=psx,
    sorted_index_method='normalized_margin',  # Orders label errors
)

print('orderd_label_errors:', ordered_label_errors)

print(np.array(sorted(ordered_label_errors)))
idx_errors = ordered_label_errors

label_errors_idx = np.array(sorted(ordered_label_errors))
score = sum([e in label_errors_idx for e in actual_label_errors]) / actual_num_errors
print('% actual errors that confident learning found: {:.0%}'.format(score))
score = sum([e in actual_label_errors for e in label_errors_idx]) / len(label_errors_idx)
print('% confident learning errors that are actual errors: {:.0%}'.format(score))

# original lr f1

print('WITHOUT confident learning,', end=" ")

clf.fit(X_train, s)
pred = clf.predict(X_test)
print("dataset test f1:", round(f1_score(pred, y_test, average='micro'), 4))

print("\nNow we show improvement using cleanlab to characterize the noise")
print("and learn on the data that is (with high confidence) labeled correctly.")
print()
print('WITH confident learning (psx not given),', end=" ")
rp = LearningWithNoisyLabels(clf=clf)
rp.fit(X_train, s)
pred = rp.predict(X_test)
print("dataset test f1:", round(f1_score(pred, y_test, average='micro'), 4))

print('WITH confident learning (psx given),', end=" ")
rp.fit(X=X_train, s=s, psx=psx)
pred = rp.predict(X_test)
print("dataset test f1:", round(f1_score(pred, y_test, average='micro'), 4))

print('WITH all label right,', end=" ")
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print("dataset test f1:", round(f1_score(pred, y_test, average='micro'), 4))

print("-------------------")
rp_score = f1_score(y_test, rp.fit(X_train, s, psx=psx).predict(X_test), average='micro')
print("Logistic regression (+rankpruning):", rp_score)
clf = LogisticRegression(solver='lbfgs', multi_class='auto')
print('Fit on denoised data without re-weighting:',
      f1_score(y_test, clf.fit(X_train[~idx_errors], s[~idx_errors]).predict(X_test), average='micro'))
