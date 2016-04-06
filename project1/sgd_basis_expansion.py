from os.path import join
import time
import hickle as hkl
from sklearn import linear_model
import cPickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing

DATA_DIR = '/home/smile/edzhou/Thesis/data'
y = hkl.load(join(DATA_DIR, 'hkl_train/train_y_gzip.hkl'))
print 'Loaded y'

N = len(y)
train_proportion = 0.8

t = 7
X_expand = hkl.load(join(DATA_DIR, 'expand_poly3_t-7_gzip.hkl'))
print 'Loaded t-7 expansion'
y_t = y[t:, ]

train_ind = np.array(xrange(0, int(train_proportion * N) - t))
test_ind = np.array(xrange(int(train_proportion * N) - t, len(y_t)))

X_train = X_expand[train_ind, :]
y_train_mat = y_t[train_ind, :]
X_test = X_expand[test_ind, :]
y_test_mat = y_t[test_ind, :]

del X_expand # save memory

scaler = preprocessing.StandardScaler(copy=False)
scaler.fit(X_train)
scaler.transform(X_train, copy=False)
scaler.transform(X_test)
print 'Finished scaling + transforming'

y_pred_mat = np.zeros(y_test_mat.shape)
N_ITER = 25
alphas = [160] * 10

RMSEs = [None] * len(alphas)
for i, alpha in enumerate(alphas):
    print 'Starting alpha = {0}'.format(alpha)
    start = time.time()
    for a in xrange(0, y.shape[1]):
        y_train = y_train_mat[:, a:(a + 1)].ravel()
        sgd = linear_model.SGDRegressor(alpha=alpha, n_iter=N_ITER)
        sgd.fit(X_train, y_train)
        y_pred_mat[:, a] = sgd.predict(X_test)
        print a
    RMSE = mean_squared_error(y_test_mat, y_pred_mat) ** 0.5
    end = time.time()
    print 'alpha = {0}, took {1} to 80/20 fit + RMSE'.format(
        alpha, end - start)
    print 'Achieved RMSE = {0}'.format(RMSE)
    print '=========='
    RMSEs[i] = RMSE

    df = pd.DataFrame({'alpha': alphas, 'RMSE': RMSEs})
    df.to_csv('sgd_basis_expanded_RMSEs_4.csv', index=False)