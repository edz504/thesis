from os.path import join
import time
import hickle as hkl
from sklearn import linear_model
import cPickle
import numpy as np
import pandas as pd
from sklearn import preprocessing
import sys

DATA_DIR = '/home/smile/edzhou/Thesis/data'
y = hkl.load(join(DATA_DIR, 'hkl_train/train_y_gzip.hkl'))
AIC_vec = [None] * 13
BIC_vec = [None] * 13
t_vec = xrange(0, 13)

for t in t_vec:
    X_t = hkl.load(
        join(DATA_DIR, 'train_cnn_500_t{0}_gzip.hkl'.format(t)))
    print 'Loaded X for t-{0}'.format(t)
    preprocessing.scale(X_t, copy=False)
    print 'Scaled X for t-{0}'.format(t)

    logL_sum = 0
    N_sum = 0
    k_sum = 0

    start = time.time()
    for a in xrange(0, y.shape[1]):
        y_true = y[t:, a:(a + 1)].ravel()
        sgd = linear_model.SGDRegressor(alpha=100, n_iter=20)
        sgd.fit(X_t, y_true)
        with open(join(DATA_DIR, 'sklearn_sgd_t{0}_a{1}.pkl'.format(t, a)),
                  'wb') as f:
            cPickle.dump(sgd, f)
        print 'Dumped t-{0}, a{1}'.format(t, a)
        y_pred = sgd.predict(X_t)
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

    print 't-{0} took {1}'.format(t, end - start)

    AIC_vec[t] = -2 * logL_sum + 2 * k_sum
    BIC_vec[t] = -2 * logL_sum + (np.log(N_sum) * k_sum)

    print 'Stacked AIC = {0} for t-{1}'.format(AIC_vec[t], t)
    print 'Stacked BIC = {0} for t-{1}'.format(BIC_vec[t], t)

    df = pd.DataFrame({'t': t_vec, 'AIC': AIC_vec, 'BIC': BIC_vec})
    df.to_csv('stacked_aic_bic_sgd2.csv', index=False)
