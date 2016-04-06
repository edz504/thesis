from os.path import join
import time
import hickle as hkl
from sklearn import linear_model
import cPickle
import numpy as np
from sklearn import preprocessing
import sys

DATA_DIR = '/home/smile/edzhou/Thesis/data'
y = hkl.load(join(DATA_DIR, 'hkl_train/train_y_gzip.hkl'))

X_cnn_0 = hkl.load(join(DATA_DIR, 'train_cnn_500_t0_gzip.hkl'))
X_cnn_1 = hkl.load(join(DATA_DIR, 'train_cnn_500_t1_gzip.hkl'))
X_cnn_2 = hkl.load(join(DATA_DIR, 'train_cnn_500_t2_gzip.hkl'))

for t, X in enumerate([X_cnn_0, X_cnn_1, X_cnn_2]):
    X_scaled = preprocessing.scale(X)
    logL_sum = 0
    N_sum = 0
    k_sum = 0
    start = time.time()
    for a in xrange(0, y.shape[1]):
        y_true = y[t:, a:(a + 1)].ravel()
        sgd = linear_model.SGDRegressor(alpha=10)
        sgd.fit(X_scaled, y_true)
        y_pred = sgd.predict(X_scaled)
        SSR = sum([(y_p - y_t) ** 2 for y_p, y_t in zip(y_pred, y_true)])
        N = y_pred.shape[0]
        s2 = SSR / N
        logL = -N * 0.5 * np.log(2 * np.pi * s2) - SSR / (2 * s2)
        k = sum([1 for coef in sgd.coef_ if abs(coef - 0.0) > 1e-10])
        logL_sum += logL
        N_sum += N
        k_sum += k
        print a
        sys.stdout.flush()
    end = time.time()
    print 't-{0}, {1} to fit'.format(t, end - start)
    print 'Stacked AIC = {0} for t = {1}'.format(-2 * logL_sum + 2 * k_sum, t)
    print 'Stacked BIC = {0} for t = {1}'.format(-2 * logL_sum + (np.log(N_sum) * k_sum), t)


# t = 1
# X = X_cnn_1
# X_scaled = preprocessing.scale(X)
# a = 0
# y_true = y[t:, a:(a + 1)].ravel()

# sgd = linear_model.SGDRegressor(alpha=10)
# sgd.fit(X_scaled, y_true)
# y_pred = sgd.predict(X_scaled)
# SSR = sum([(y_p - y_t) ** 2 for y_p, y_t in zip(y_pred, y_true)])
# N = y_pred.shape[0]
# s2 = SSR / N
# logL = -N * 0.5 * np.log(2 * np.pi * s2) - SSR / (2 * s2)
# k = sum([1 for coef in sgd.coef_ if abs(coef - 0.0) > 1e-10])
# print 2 * k - 2 * logL