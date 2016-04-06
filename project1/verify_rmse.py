from os.path import join
import time
import hickle as hkl
from sklearn import linear_model
import cPickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

DATA_DIR = '/home/smile/edzhou/Thesis/data'
y_scaled = hkl.load(join(DATA_DIR, 'train_y_scaled_gzip.hkl'))

T = 7
N = len(y_scaled)
DATA_DIR = '/home/smile/edzhou/Thesis/data'
y_scaled = hkl.load(join(DATA_DIR, 'train_y_scaled_gzip.hkl'))
train_proportion = 0.8
all_ind = np.array(xrange(0, N))

with open('train_inds.txt', 'rb') as f:
    train_ind = np.array([int(i)
                           for i in f.read().split('\n') if i != ''])

test_ind = np.array(list(set(all_ind) - set(train_ind)))

t_vec = xrange(0, 8)
RMSE_vec = [None] * len(t_vec)
for t in t_vec:
    X_cnn_this_t = hkl.load(
        join(DATA_DIR, 'train_cnn_500_t{0}_gzip.hkl'.format(t)))
    print 'Loaded X for t-{0}'.format(t)

    # Split into test and train, making sure to adjust the indices for
    # out of bounds.  Note that each iteration of t removes one index.
    # See confirm_split_train_test.py for example.
    if t > 0:
        train_ind = [ind - 1 for ind in train_ind]
        train_ind.remove(-1)
        train_ind = np.array(train_ind)
        test_ind = [ind - 1 for ind in test_ind]

    X_train = X_cnn_this_t[train_ind, :]
    y_train = y_scaled[train_ind, :]

    X_test = X_cnn_this_t[test_ind, :]
    y_test = y_scaled[test_ind, :]

    # Fit on training data.
    start = time.time()
    regr = linear_model.LinearRegression(n_jobs = -1) # use all CPUs
    regr.fit(X_train, y_train)
    end = time.time()
    print 't-{0} took {1} to fit'.format(t, end - start)

    # Calculate RMSE.
    y_pred_mat = regr.predict(X_test)
    RMSE = mean_squared_error(y_test, y_pred_mat) ** 0.5
    print RMSE

    logL_sum = 0
    N_sum = 0
    k_sum = 0

    # Verify AIC and BIC are logical
    y_pred_mat = regr.predict(X_train)
    for col in xrange(0, y_test.shape[1]):
        y_pred = y_pred_mat[:, col]
        y_true = y_test[:, col]
        these_coef = regr.coef_[col, :]
        SSR = sum([(y_p - y_t) ** 2 for y_p, y_t in zip(y_pred, y_true)])
        print SSR
        N = y_pred.shape[0]
        s2 = SSR / N
        logL = -N * 0.5 * np.log(2 * np.pi * s2) - SSR / (2 * s2)
        print logL
        k = sum([1 for coef in these_coef if abs(coef - 0.0) > 1e-10])
        logL_sum += logL
        N_sum += N
        k_sum += k
        print 'AIC for a{0}, t-{1} = {2}'.format(col, t, 2 * k - 2 * logL)
        print 'BIC for a{0}, t-{1} = {2}'.format(col, t, np.log(N_sum) * k - 2 * logL)
    print 'Stacked AIC: {0}'.format(-2 * logL_sum + 2 * k_sum)
    print 'Stacked BIC: {0}'.format(-2 * logL_sum + np.log(N_sum) * k_sum)