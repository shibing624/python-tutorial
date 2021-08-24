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



# To silence convergence warnings caused by using a weak
# logistic regression classifier on image data
import warnings

import cleanlab
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression

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

# Add lots of errors to labels
s = np.array(y)
for i in range(100):
    # Switch to some wrong label thats a different class
    s[i] = 0

# Confirm that we indeed added NUM_ERRORS label errors
actual_label_errors = np.arange(len(y))[s != y]
print('\nIndices of actual label errors:\n', actual_label_errors)
print('error with y, y[:20]:', s[:20])
print("len of errors:", len(actual_label_errors))
actual_num_errors = len(actual_label_errors)
# To keep the tutorial short, we use cleanlab to get the
# out-of-sample predicted probabilities using cross-validation
# with a very simple, non-optimized logistic regression classifier
psx = cleanlab.latent_estimation.estimate_cv_predicted_probabilities(
    X, s, clf=LogisticRegression(solver='lbfgs'))

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
    frac_noise=2.1,
    sorted_index_method='normalized_margin',  # Orders label errors
)

print('orderd_label_errors:', ordered_label_errors)

print(np.array(sorted(ordered_label_errors)))


label_errors_idx = np.array(sorted(ordered_label_errors))
score = sum([e in label_errors_idx for e in actual_label_errors]) / actual_num_errors
print('% actual errors that confident learning found: {:.0%}'.format(score))
score = sum([e in actual_label_errors for e in label_errors_idx]) / len(label_errors_idx)
print('% confident learning errors that are actual errors: {:.0%}'.format(score))
